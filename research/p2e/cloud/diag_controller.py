#!/usr/bin/env python3
"""Autonomous freeze-diagnosis + self-healing experiment runner.
Runs short variants to (a) CAPTURE the exact stack of the freeze, (b) ISOLATE which
component causes it, then launches the most-complete WORKING config as the real
experiment under a hang-aware watchdog. Writes everything to /workspace/DIAG.md."""
import os, sys, time, signal, subprocess, glob, re, datetime

PROJ="/workspace/proj"; RUN=[]
DIAG="/workspace/DIAG.md"; STATUS="/workspace/STATUS.txt"
ENVBASE=dict(os.environ, PYTHONUNBUFFERED="1", PYTHONFAULTHANDLER="1",
    PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128",
    PYTHONPATH=f"{PROJ}:{PROJ}/research/p2e/world:{PROJ}/research/p2e/retrieval:{PROJ}/research/p2e/drive:{PROJ}/research/p2e/monitor",
    MEM_K="4", LP_SUBSAMPLE="128", MEM_DRIVE_MAX="50000", MEM_INJECT_MAX="50000")

def now(): return datetime.datetime.utcnow().strftime("%H:%M:%S")
def log(m):
    open(DIAG,"a").write(m+"\n"); open(DIAG,"a").flush(); print(m,flush=True)
def setstatus(m): open(STATUS,"w").write(f"{now()} {m}\n")

def sheeprl_cmd(mem_aug, total=1300):
    mlp = "algo.mlp_keys.encoder=[memory]" if mem_aug else "algo.mlp_keys.encoder=[]"
    return RUN+["sheeprl","exp=p2e_dv3_exploration","env=crafter",
        "env.wrapper._target_=crafter_tvs.CrafterTVWrapper","fabric.accelerator=cuda",
        "env.num_envs=1","env.sync_env=True","algo.cnn_keys.encoder=[rgb]",
        "algo.cnn_keys.decoder=[rgb]",mlp,"algo.mlp_keys.decoder=[]",
        "algo.per_rank_batch_size=16","algo.per_rank_sequence_length=64","buffer.size=20000",
        "algo.run_test=False","metric.log_level=1","metric.log_every=400",
        "checkpoint.save_last=False","model_manager.disabled=True",
        f"algo.total_steps={total}","algo.learning_starts=1024"]

def poll_step(logf):
    try: t=open(logf).read()
    except: return 0, False
    steps=re.findall(r"policy_step=(\d+)", t)
    trained = "world_model_loss" in t or "Loss/world_model" in t
    return (int(steps[-1]) if steps else 0), trained

def run_variant(name, extra_env, mem_aug):
    """Returns ('RUNS'|'HUNG'|'CRASHED', last_step, stack_text)."""
    logf=f"/workspace/var_{name}.log"
    for f in glob.glob(f"{PROJ}/logs/runs/**",recursive=True): pass
    os.system(f"rm -rf {PROJ}/logs")
    env=dict(ENVBASE, **extra_env)
    log(f"\n### VARIANT `{name}`  ({', '.join(k+'='+v for k,v in extra_env.items())})  {now()}")
    setstatus(f"running variant {name}")
    fo=open(logf,"w")
    p=subprocess.Popen(sheeprl_cmd(mem_aug), cwd=PROJ, env=env, stdout=fo, stderr=subprocess.STDOUT)
    last=0; stuck=0; outcome=None
    for i in range(60):  # up to ~10 min
        time.sleep(10)
        if p.poll() is not None:
            txt=open(logf).read()
            outcome = "CRASHED" if re.search(r"Traceback|Error executing|RuntimeError", txt) else "EXITED"
            break
        step, trained = poll_step(logf)
        if trained or step>1250:
            outcome="RUNS"; last=step; break
        if step<=last: stuck+=1
        else: stuck=0; last=step
        if stuck>=8:  # ~80s no progress past learning_starts -> HUNG: dump stack
            log(f"  {name} HUNG at step {step} — sending SIGUSR1 for stack dump")
            pids=subprocess.run(["pgrep","-f","sheeprl"],capture_output=True,text=True).stdout.split()
            for pid in pids:
                try: os.kill(int(pid), signal.SIGUSR1)
                except: pass
            time.sleep(6)
            outcome="HUNG"; break
    else:
        outcome="RUNS"; last,_=poll_step(logf)
    # capture stack (faulthandler dump lands in the log on SIGUSR1)
    txt=open(logf).read()
    stack=""
    m=re.split(r"(Thread 0x|Current thread 0x)", txt)
    if len(m)>1: stack="".join(m[-3:])[-3000:]
    try: p.send_signal(signal.SIGKILL)
    except: pass
    os.system("pkill -9 -f algos.p2e 2>/dev/null; pkill -9 -f 'sheeprl exp' 2>/dev/null"); time.sleep(3)
    log(f"  => OUTCOME: **{outcome}** (last policy_step {last})")
    if outcome=="HUNG" and stack:
        log("  <details><summary>frozen stack</summary>\n\n```\n"+stack+"\n```\n</details>")
    return outcome, last, stack

def watch_and_heal(cmd, env, max_restarts=4):
    """Run the real experiment; on a hang dump the stack (SIGUSR1) THEN restart; cap restarts."""
    restarts=0
    while True:
        log(f"  [exp] launch attempt {restarts} {now()}")
        fo=open("/workspace/experiment.log","a")
        p=subprocess.Popen(cmd, cwd=PROJ, env=env, stdout=fo, stderr=subprocess.STDOUT)
        last=0; stuck=0
        while True:
            time.sleep(30)
            if p.poll() is not None:
                step,_=poll_step("/workspace/experiment.log")
                log(f"  [exp] process exited at step {step}")
                if step>=99000: log("  [exp] COMPLETE"); setstatus("EXPERIMENT COMPLETE"); return
                setstatus(f"exp exited at {step}, restarting"); break
            step,_=poll_step("/workspace/experiment.log")
            if step<=last: stuck+=1
            else: stuck=0; last=step
            setstatus(f"EXPERIMENT training, step {step}")
            if stuck>=10:  # ~5 min no progress -> hang
                log(f"  [exp] HANG at step {step} -> SIGUSR1 stack dump + restart")
                for pid in subprocess.run(["pgrep","-f","sheeprl"],capture_output=True,text=True).stdout.split():
                    try: os.kill(int(pid),signal.SIGUSR1)
                    except: pass
                time.sleep(6); os.system(f"cp /workspace/experiment.log /workspace/experiment_hang_{restarts}.log")
                try: p.send_signal(signal.SIGKILL)
                except: pass
                os.system("pkill -9 -f algos.p2e 2>/dev/null"); time.sleep(3); break
        restarts+=1
        if restarts>max_restarts:
            log("  [exp] gave up after repeated hangs (stacks saved as experiment_hang_*.log)")
            setstatus("EXPERIMENT gave up - persistent hang"); return

def main():
    open(DIAG,"w").write(f"# Freeze diagnosis — {datetime.datetime.utcnow().isoformat()}Z\n")
    log("Goal: capture the EXACT frozen stack at the first training step and isolate the cause.\n")
    # Order: failing config FIRST (capture the stack), then isolate by toggling each suspect.
    variants=[
        ("v0_full",       dict(MEM_AUG="1",DRIVE="lp",MEM_ABLATE="1",MEM_ABLATE_EVERY="16"), True),
        ("v1_no_ablate",  dict(MEM_AUG="1",DRIVE="lp"), True),
        ("v2_ens_aug",    dict(MEM_AUG="1",DRIVE="ensemble"), True),
        ("v3_lp_no_aug",  dict(MEM_AUG="0",DRIVE="lp"), False),
        ("v4_stock",      dict(MEM_AUG="0",DRIVE="ensemble"), False),
    ]
    results={}
    for name,env,aug in variants:
        try: results[name]=run_variant(name,env,aug)
        except Exception as e: log(f"  controller error on {name}: {e!r}"); results[name]=("ERROR",0,"")
    # conclusion
    log("\n## CONCLUSION")
    for n,(o,s,_) in results.items(): log(f"- {n}: {o} (step {s})")
    runs=[n for n,(o,_,_) in results.items() if o=="RUNS"]
    log(f"\nConfigs that PASSED the freeze: {runs or 'NONE'}")
    if "v0_full" in results and results["v0_full"][0]=="HUNG":
        # infer culprit from which toggle fixed it
        if results.get("v2_ens_aug",("",))[0]=="RUNS": log("**Cause: the learning-progress DRIVE (faiss `batch_reward` inside train()) — DRIVE=ensemble runs, DRIVE=lp hangs.**")
        elif results.get("v1_no_ablate",("",))[0]=="RUNS": log("**Cause: the MEM_ABLATE extra world-model passes.**")
        elif results.get("v3_lp_no_aug",("",))[0]=="RUNS": log("**Cause: the memory INJECTOR (MEM_AUG).**")
        else: log("**Cause: see the captured stack above (not isolated by the toggles).**")
    # launch the most-complete working config as the real experiment under a hang-aware watchdog
    pref=["v0_full","v1_no_ablate","v3_lp_no_aug","v2_ens_aug","v4_stock"]
    winner=next((n for n in pref if results.get(n,("",))[0]=="RUNS"), None)
    if winner:
        log(f"\n## LAUNCHING REAL EXPERIMENT with `{winner}` (passed the freeze) under hang-aware watchdog")
        setstatus(f"EXPERIMENT running ({winner})")
        env=dict(ENVBASE, **dict(variants[[v[0] for v in variants].index(winner)][1]))
        os.system(f"rm -rf {PROJ}/logs")
        fo=open("/workspace/experiment.log","w")
        cmd=[c for c in sheeprl_cmd(variants[[v[0] for v in variants].index(winner)][2], total=100000)]
        # bump logging + checkpoints for the real run
        cmd=[re.sub(r"total_steps=1300","total_steps=100000",c) for c in cmd]
        watch_and_heal(cmd, env)
    else:
        log("\n## NO config passed the freeze — leaving the captured stacks for diagnosis. NOT burning more time blindly.")
        setstatus("ALL VARIANTS FAILED - see DIAG.md stacks")
    log(f"\n_diagnostic loop done {now()}_")

main()
