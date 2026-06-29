"""Apply the full integration (memory-augmentation + learning-progress drive) to a
fresh sheeprl install.

The sheeprl package lives in a gitignored venv, so the edits can't be committed
directly. This script re-applies them by exact string replacement. Idempotent: each
phase skips files it has already patched. Run after `pip install sheeprl` (e.g. on
the cloud box):

    python research/p2e/patches/apply_integration.py

Two phases:
  1. MEMORY-AUG (dreamer_v3.py + p2e_dv3_exploration.py): add a fixed-size "memory"
     encoder key fed from the FaissNetworkXStore graph (env MEM_AUG=1).
  2. DRIVE-SWAP (p2e_dv3_exploration.py only): replace ensemble-disagreement intrinsic
     reward with a learning-progress reward model trained on memory-computed targets
     (env DRIVE=lp; DRIVE=ensemble keeps stock Plan2Explore for the baseline arm).

Both verified to TRAIN END-TO-END on Mac/CPU (seeded Crafter + TV tiles):
intrinsic (LP) reward falls 0.45 -> 0.15 as the world model is learned, lp regression
loss 4.63 -> 0.90, world_model_loss 770 -> 124, no errors. See research/p2e/dryrun_lp.sh.

Launch (combined) with env:
    KMP_DUPLICATE_LIB_OK=TRUE MEM_AUG=1 DRIVE=lp
    PYTHONPATH=research/p2e/world:research/p2e/retrieval:research/p2e/drive:.
and hydra overrides 'algo.mlp_keys.encoder=[memory]' 'algo.mlp_keys.decoder=[]'
env.sync_env=True algo.run_test=False (see research/p2e/dryrun_lp.sh for the full line).
"""

from __future__ import annotations

import importlib.util
import os

# ---------------------------------------------------------------------------
# Phase 1: memory-augmentation (shared by dreamer_v3.py and p2e_dv3_exploration.py)
# ---------------------------------------------------------------------------

_IMPORT_BLOCK = '''
# --- memory-augmentation wire-in (research/p2e) ---
import os as _mem_os
import sys as _mem_sys
_mem_sys.path.insert(0, _mem_os.environ.get("MEM_INJECT_PATH", os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..", "..", "research", "p2e", "retrieval")))
from inject import MemoryInjector  # noqa: E402'''

_OBSSPACE_BLOCK = '''

    # --- memory-augmentation: add a fixed-size "memory" key to the obs space ---
    _MEM_ON = _mem_os.environ.get("MEM_AUG", "0") == "1"
    _MEM_K = int(_mem_os.environ.get("MEM_K", "4"))
    _lat_dim = (
        cfg.algo.world_model.stochastic_size * cfg.algo.world_model.discrete_size
        + cfg.algo.world_model.recurrent_model.recurrent_state_size
    )
    _MEM_DIM = _MEM_K * _lat_dim
    _injector = None
    if _MEM_ON:
        observation_space = gym.spaces.Dict(
            {**observation_space.spaces,
             "memory": gym.spaces.Box(-np.inf, np.inf, (_MEM_DIM,), np.float32)}
        )
        _injector = MemoryInjector(latent_dim=_lat_dim, k=_MEM_K, n_envs=cfg.env.num_envs)'''

_STEPDATA_BLOCK = '''    step_data["is_first"] = np.ones_like(step_data["terminated"])
    if _MEM_ON:
        step_data["memory"] = np.zeros((1, cfg.env.num_envs, _MEM_DIM), dtype=np.float32)
    player.init_states()'''

_COLLECT_BLOCK = '''                    torch_obs = prepare_obs(fabric, obs, cnn_keys=cfg.algo.cnn_keys.encoder, num_envs=cfg.env.num_envs)
                    if _MEM_ON:
                        _lat = torch.cat((player.stochastic_state, player.recurrent_state), -1)
                        _lat = _lat.reshape(cfg.env.num_envs, _lat_dim).detach().cpu().numpy()
                        _mem = _injector.step(_lat).astype(np.float32)
                        step_data["memory"] = _mem[None]
                        torch_obs["memory"] = torch.as_tensor(_mem, device=fabric.device).float().unsqueeze(0)
                    mask = {k: v for k, v in torch_obs.items() if k.startswith("mask")}'''

_MEM_EDITS = [
    ("    obs_keys = cfg.algo.cnn_keys.encoder + cfg.algo.mlp_keys.encoder",
     '    obs_keys = [k for k in (cfg.algo.cnn_keys.encoder + cfg.algo.mlp_keys.encoder) if k != "memory"]'),
    ("    observation_space = envs.single_observation_space",
     "    observation_space = envs.single_observation_space" + _OBSSPACE_BLOCK),
    ('''    step_data["is_first"] = np.ones_like(step_data["terminated"])
    player.init_states()''', _STEPDATA_BLOCK),
    ('''                    torch_obs = prepare_obs(fabric, obs, cnn_keys=cfg.algo.cnn_keys.encoder, num_envs=cfg.env.num_envs)
                    mask = {k: v for k, v in torch_obs.items() if k.startswith("mask")}''', _COLLECT_BLOCK),
]

_IMPORT_ANCHORS = {
    "dreamer_v3.py": "from sheeprl.algos.dreamer_v3.agent import WorldModel, build_agent",
    "p2e_dv3_exploration.py": "from sheeprl.algos.p2e_dv3.agent import build_agent",
}

# ---------------------------------------------------------------------------
# Phase 2: learning-progress drive swap (p2e_dv3_exploration.py only)
# ---------------------------------------------------------------------------

_DRIVE_FLAG_OLD = "from inject import MemoryInjector  # noqa: E402"
_DRIVE_FLAG_NEW = '''from inject import MemoryInjector  # noqa: E402

# --- learning-progress drive wire-in (research/p2e) ---
# DRIVE=lp  -> intrinsic reward = learning progress from the real memory graph.
# DRIVE=ensemble (default) -> stock Plan2Explore ensemble disagreement (baseline arm).
_DRIVE_LP = _mem_os.environ.get("DRIVE", "ensemble").lower() == "lp"
_mem_sys.path.insert(0, _mem_os.environ.get("DRIVE_PATH", os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..", "..", "research", "p2e", "drive")))
_LP: dict = {}  # populated in main() when _DRIVE_LP: {"model","opt","drive","t"}'''

_LP_HELPER = '''def _train_lp_reward(fabric, po, batch_obs, latent_states, device):
    """Train the learning-progress reward model on the REAL batch.

    For every real latent state in the batch, the per-step world-model prediction
    error (observation loss) is fed to the memory graph (MemoryDrive), which stores
    it and returns the learning-progress reward (slope of error-over-time for similar
    past states). The lp_reward_model is then regressed (two-hot, like the reward
    model) to predict that target from the latent — so it generalises the
    memory-computed drive to the imagined latents used during behaviour learning.

    Returns the LP regression loss (logged in place of the ensemble loss).
    """
    lpm = _LP["model"]
    lpo = _LP["opt"]
    drive = _LP["drive"]
    # per-[T, B] prediction error = the world model's per-step observation loss
    # (same quantity loss.py reduces to observation_loss.mean()).
    per_step_err = (-sum(po[k].log_prob(batch_obs[k]) for k in po.keys())).detach()
    T, B = per_step_err.shape[0], per_step_err.shape[1]
    # subsample the batch's latents for the LP targets (faiss work scales with this count);
    # the reward model still sees plenty of (latent, target) pairs across training steps.
    flat = latent_states.detach().reshape(T * B, -1)
    err = per_step_err.reshape(T * B)
    _sub = min(int(_mem_os.environ.get("LP_SUBSAMPLE", "128")), T * B)
    _sel = torch.randperm(T * B, device=flat.device)[:_sub]
    lat_s = flat[_sel]
    tgt = drive.batch_reward(lat_s.cpu().numpy(), err[_sel].cpu().numpy(), step=float(_LP["t"]))
    _LP["t"] += 1
    targets = torch.as_tensor(tgt, device=device).reshape(_sub, 1)
    lpo.zero_grad(set_to_none=True)
    lp_dist = TwoHotEncodingDistribution(lpm(lat_s), dims=1)
    lp_loss = -lp_dist.log_prob(targets).mean()
    fabric.backward(lp_loss)
    lpo.step()
    return lp_loss.detach()


def train(
    fabric: Fabric,
    world_model: WorldModel,'''

_ENSEMBLE_OLD = '''    # Ensemble Learning
    loss = 0.0
    ensemble_optimizer.zero_grad(set_to_none=True)
    for ens in ensembles:
        out = ens(
            torch.cat(
                (
                    posteriors.view(*posteriors.shape[:-2], -1).detach(),
                    recurrent_states.detach(),
                    data["actions"].detach(),
                ),
                -1,
            )
        )[:-1]
        next_state_embedding_dist = MSEDistribution(out, 1)
        loss -= next_state_embedding_dist.log_prob(posteriors.view(sequence_length, batch_size, -1).detach()[1:]).mean()
    loss.backward()
    ensemble_grad = None
    if cfg.algo.ensembles.clip_gradients is not None and cfg.algo.ensembles.clip_gradients > 0:
        ensemble_grad = fabric.clip_gradients(
            module=ens,
            optimizer=ensemble_optimizer,
            max_norm=cfg.algo.ensembles.clip_gradients,
            error_if_nonfinite=False,
        )
    ensemble_optimizer.step()'''

_ENSEMBLE_NEW = '''    # Ensemble Learning  (DRIVE=lp swaps this for learning-progress reward learning)
    ensemble_grad = None
    if _DRIVE_LP:
        # Train the learning-progress reward model on the real batch instead of the
        # ensemble. `loss` carries the LP regression loss for the metric logger.
        loss = _train_lp_reward(fabric, po, batch_obs, latent_states, device)
    else:
        loss = 0.0
        ensemble_optimizer.zero_grad(set_to_none=True)
        for ens in ensembles:
            out = ens(
                torch.cat(
                    (
                        posteriors.view(*posteriors.shape[:-2], -1).detach(),
                        recurrent_states.detach(),
                        data["actions"].detach(),
                    ),
                    -1,
                )
            )[:-1]
            next_state_embedding_dist = MSEDistribution(out, 1)
            loss -= next_state_embedding_dist.log_prob(
                posteriors.view(sequence_length, batch_size, -1).detach()[1:]
            ).mean()
        loss.backward()
        if cfg.algo.ensembles.clip_gradients is not None and cfg.algo.ensembles.clip_gradients > 0:
            ensemble_grad = fabric.clip_gradients(
                module=ens,
                optimizer=ensemble_optimizer,
                max_norm=cfg.algo.ensembles.clip_gradients,
                error_if_nonfinite=False,
            )
        ensemble_optimizer.step()'''

_REWARD_OLD = '''        if critic["reward_type"] == "intrinsic":
            # Predict intrinsic reward
            next_state_embedding = torch.empty(
                len(ensembles),
                cfg.algo.horizon + 1,
                batch_size * sequence_length,
                stochastic_size * discrete_size,
                device=device,
            )
            for i, ens in enumerate(ensembles):
                next_state_embedding[i] = ens(
                    torch.cat((imagined_trajectories.detach(), imagined_actions.detach()), -1)
                )

            # next_state_embedding -> N_ensemble x Horizon x Batch_size*Seq_len x Obs_embedding_size
            reward = next_state_embedding.var(0).mean(-1, keepdim=True) * cfg.algo.intrinsic_reward_multiplier
            if aggregator and not aggregator.disabled:
                aggregator.update(f"Rewards/intrinsic_{k}", reward.detach().cpu().mean())'''

_REWARD_NEW = '''        if critic["reward_type"] == "intrinsic":
            if _DRIVE_LP:
                # Learning-progress drive: the trained lp_reward_model predicts the
                # memory-computed reward for each imagined latent (mirrors how the task
                # reward_model is queried over imagined trajectories).
                reward = (
                    TwoHotEncodingDistribution(_LP["model"](imagined_trajectories), dims=1).mean
                    * cfg.algo.intrinsic_reward_multiplier
                )
            else:
                # Predict intrinsic reward
                next_state_embedding = torch.empty(
                    len(ensembles),
                    cfg.algo.horizon + 1,
                    batch_size * sequence_length,
                    stochastic_size * discrete_size,
                    device=device,
                )
                for i, ens in enumerate(ensembles):
                    next_state_embedding[i] = ens(
                        torch.cat((imagined_trajectories.detach(), imagined_actions.detach()), -1)
                    )

                # next_state_embedding -> N_ensemble x Horizon x Batch_size*Seq_len x Obs_embedding_size
                reward = next_state_embedding.var(0).mean(-1, keepdim=True) * cfg.algo.intrinsic_reward_multiplier
            if aggregator and not aggregator.disabled:
                aggregator.update(f"Rewards/intrinsic_{k}", reward.detach().cpu().mean())'''

_LP_BUILD_OLD = '''    for k, critic in critics_exploration.items():
        critic["optimizer"] = fabric.setup_optimizers(critic["optimizer"])

    moments_exploration = {'''

_LP_BUILD_NEW = '''    for k, critic in critics_exploration.items():
        critic["optimizer"] = fabric.setup_optimizers(critic["optimizer"])

    # --- learning-progress drive: reward model + optimizer + memory graph ---
    if _DRIVE_LP:
        from sheeprl.models.models import MLP as _LPMLP
        from sheeprl.algos.dreamer_v3.utils import init_weights as _lp_init
        from sheeprl.algos.dreamer_v3.utils import uniform_init_weights as _lp_uinit

        _lp_ln_cls = hydra.utils.get_class(cfg.algo.critic.layer_norm.cls)
        _lp_model = _LPMLP(
            input_dims=_lat_dim,
            output_dim=cfg.algo.critic.bins,
            hidden_sizes=[cfg.algo.critic.dense_units] * cfg.algo.critic.mlp_layers,
            activation=hydra.utils.get_class(cfg.algo.critic.dense_act),
            flatten_dim=None,
            layer_args={"bias": _lp_ln_cls == nn.Identity},
            norm_layer=_lp_ln_cls,
            norm_args={**cfg.algo.critic.layer_norm.kw, "normalized_shape": cfg.algo.critic.dense_units},
        )
        _lp_model.apply(_lp_init)
        if cfg.algo.hafner_initialization:
            _lp_model.model[-1].apply(_lp_uinit(0.0))
        _lp_model = fabric.setup_module(_lp_model)
        _lp_opt = hydra.utils.instantiate(cfg.algo.critic.optimizer, params=_lp_model.parameters(), _convert_="all")
        _lp_opt = fabric.setup_optimizers(_lp_opt)
        from lp_memory_reward import MemoryDrive as _MemoryDrive

        _LP["model"] = _lp_model
        _LP["opt"] = _lp_opt
        _LP["drive"] = _MemoryDrive()
        _LP["t"] = 0
        # same-run resume: reload the drive's memory graph saved by a previous segment
        _drive_load = _mem_os.environ.get("DRIVE_LOAD", "")
        if _drive_load and os.path.exists(_drive_load):
            _LP["drive"].load_store(_drive_load)
            fabric.print(f"[DRIVE_LOAD] reloaded drive memory: {_LP['drive'].node_count()} nodes")
        fabric.print(f"[DRIVE=lp] learning-progress reward model built (latent_dim={_lat_dim})")

    moments_exploration = {'''

_DRIVE_EDITS = [
    (_DRIVE_FLAG_OLD, _DRIVE_FLAG_NEW),
    ("""def train(
    fabric: Fabric,
    world_model: WorldModel,""", _LP_HELPER),
    (_ENSEMBLE_OLD, _ENSEMBLE_NEW),
    (_REWARD_OLD, _REWARD_NEW),
    (_LP_BUILD_OLD, _LP_BUILD_NEW),
]

# ---------------------------------------------------------------------------
# Phase 3: experiment scoreboard (TV noise-trap behaviour; p2e only)
# ---------------------------------------------------------------------------

_SB_FLAG_OLD = '_LP: dict = {}  # populated in main() when _DRIVE_LP: {"model","opt","drive","t"}'
_SB_FLAG_NEW = '''_LP: dict = {}  # populated in main() when _DRIVE_LP: {"model","opt","drive","t"}

# torch>=2.6 defaults torch.load(weights_only=True), which rejects sheeprl's full
# checkpoint (replay buffer + configs, not just tensors) on resume. Our checkpoints are
# local and trusted, so restore the old full-load behavior.
_torch_orig_load = torch.load
def _torch_load_full(*_a, **_k):
    _k.setdefault("weights_only", False)
    return _torch_orig_load(*_a, **_k)
torch.load = _torch_load_full

# --- experiment scoreboard (TV noise-trap behaviour) ---
_SCOREBOARD_ON = _mem_os.environ.get("SCOREBOARD", "0") == "1"
_mem_sys.path.insert(0, _mem_os.environ.get("MONITOR_PATH", os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..", "..", "research", "p2e", "monitor")))
_SB: dict = {}  # populated in main() when _SCOREBOARD_ON: {"board": Scoreboard}'''

_SB_CREATE_OLD = '''    if _MEM_ON:
        step_data["memory"] = np.zeros((1, cfg.env.num_envs, _MEM_DIM), dtype=np.float32)
    player.init_states()'''
_SB_CREATE_NEW = '''    if _MEM_ON:
        step_data["memory"] = np.zeros((1, cfg.env.num_envs, _MEM_DIM), dtype=np.float32)
    player.init_states()

    if _SCOREBOARD_ON:
        from scoreboard import Scoreboard

        _sb_dir = _mem_os.environ.get("SCOREBOARD_DIR", os.path.join(log_dir, "scoreboard"))
        _SB["board"] = Scoreboard(arm=_mem_os.environ.get("ARM", "run"), out_dir=_sb_dir)
        fabric.print(f"[SCOREBOARD] arm={_SB['board'].arm} -> {_SB['board'].path}")'''

_SB_UPDATE_OLD = '''            step_data["is_first"] = np.zeros_like(step_data["terminated"])
            if "restart_on_exception" in infos:'''
_SB_UPDATE_NEW = '''            step_data["is_first"] = np.zeros_like(step_data["terminated"])
            if _SCOREBOARD_ON and "on_tv" in infos:
                _ot = infos["on_tv"]
                _pp = infos.get("player_pos")
                _mk = infos.get("_on_tv", np.ones(len(_ot), dtype=bool))
                for _i in range(len(_ot)):
                    if _mk[_i]:
                        _SB["board"].update(bool(_ot[_i]), _pp[_i] if _pp is not None else None)
                if _SB["board"].total_steps % 200 == 0:
                    _SB["board"].dump()
            if "restart_on_exception" in infos:'''

_SB_FINAL_OLD = '''    envs.close()
    # task test zero-shot'''
_SB_FINAL_NEW = '''    envs.close()
    if _SCOREBOARD_ON and "board" in _SB:
        _SB["board"].dump()
        from scoreboard import report

        fabric.print(report(_SB["board"].out_dir))
    # task test zero-shot'''

_SCOREBOARD_EDITS = [
    (_SB_FLAG_OLD, _SB_FLAG_NEW),
    (_SB_CREATE_OLD, _SB_CREATE_NEW),
    (_SB_UPDATE_OLD, _SB_UPDATE_NEW),
    (_SB_FINAL_OLD, _SB_FINAL_NEW),
]

# ---------------------------------------------------------------------------
# Phase 4: validation probes — (B) memory-helps-prediction ablation +
#          (C) cross-world memory transfer (save/load). p2e only.
# ---------------------------------------------------------------------------

_ABL_FLAG_OLD = '_SB: dict = {}  # populated in main() when _SCOREBOARD_ON: {"board": Scoreboard}'
_ABL_FLAG_NEW = '''_SB: dict = {}  # populated in main() when _SCOREBOARD_ON: {"board": Scoreboard}

# --- (B) memory-helps-prediction ablation ---
# Periodically recompute the world-model observation loss with the injected memory
# present vs ZEROED (same weights, same batch). delta = memory's effect on prediction.
_MEM_ABLATE = _mem_os.environ.get("MEM_ABLATE", "0") == "1"
_ABL_EVERY = int(_mem_os.environ.get("MEM_ABLATE_EVERY", "8"))
_ABL: dict = {"mem": [], "nomem": []}
_abl_counter = [0]'''

_ABL_HELPER_OLD = '''def train(
    fabric: Fabric,
    world_model: WorldModel,'''
_ABL_HELPER_NEW = '''def _world_obs_loss(world_model, batch_obs, batch_actions, is_first, cfg):
    """Mean observation (prediction) loss for a given batch_obs — used to ablate memory.

    Replicates the world-model forward (encoder -> RSSM rollout -> decoder) for the
    provided observations and returns the mean per-step observation loss. Run under
    no_grad with the current weights; called twice (memory present vs zeroed) so the
    ONLY difference is the injected memory => the delta is memory's effect on prediction.
    """
    rms = cfg.algo.world_model.recurrent_model.recurrent_state_size
    ss = cfg.algo.world_model.stochastic_size
    ds = cfg.algo.world_model.discrete_size
    seqlen = batch_actions.shape[0]
    bs = batch_actions.shape[1]
    device = batch_actions.device
    rec = torch.zeros(1, bs, rms, device=device)
    post = torch.zeros(1, bs, ss, ds, device=device)
    recs = torch.empty(seqlen, bs, rms, device=device)
    posts = torch.empty(seqlen, bs, ss, ds, device=device)
    emb = world_model.encoder(batch_obs)
    for i in range(seqlen):
        rec, post, _, _, _ = world_model.rssm.dynamic(
            post, rec, batch_actions[i : i + 1], emb[i : i + 1], is_first[i : i + 1]
        )
        recs[i] = rec
        posts[i] = post
    latent = torch.cat((posts.view(*posts.shape[:-2], -1), recs), -1)
    recon = world_model.observation_model(latent)
    po = {k: MSEDistribution(recon[k], dims=len(recon[k].shape[2:])) for k in cfg.algo.cnn_keys.decoder}
    po.update({k: SymlogDistribution(recon[k], dims=len(recon[k].shape[2:])) for k in cfg.algo.mlp_keys.decoder})
    return (-sum(po[k].log_prob(batch_obs[k]) for k in po.keys())).mean()


def train(
    fabric: Fabric,
    world_model: WorldModel,'''

_ABL_RUN_OLD = '''    world_optimizer.step()

    # Free up space
    del posterior'''
_ABL_RUN_NEW = '''    world_optimizer.step()

    # (B) memory ablation: prediction loss with memory present vs zeroed (current weights)
    if _MEM_ABLATE and "memory" in batch_obs:
        _abl_counter[0] += 1
        if _abl_counter[0] % _ABL_EVERY == 0:
            with torch.no_grad():
                _ol_mem = _world_obs_loss(world_model, batch_obs, batch_actions, data["is_first"], cfg)
                _bo0 = dict(batch_obs)
                _bo0["memory"] = torch.zeros_like(_bo0["memory"])
                _ol_nomem = _world_obs_loss(world_model, _bo0, batch_actions, data["is_first"], cfg)
            _ABL["mem"].append(float(_ol_mem))
            _ABL["nomem"].append(float(_ol_nomem))

    # Free up space
    del posterior'''

_MEMLOAD_OLD = '        _injector = MemoryInjector(latent_dim=_lat_dim, k=_MEM_K, n_envs=cfg.env.num_envs)'
_MEMLOAD_NEW = '''        _injector = MemoryInjector(latent_dim=_lat_dim, k=_MEM_K, n_envs=cfg.env.num_envs)
        # (C) transfer: carry a memory graph saved from a previous (different) world
        _mem_load = _mem_os.environ.get("MEM_LOAD", "")
        if _mem_load and os.path.exists(_mem_load):
            _injector.load_store(_mem_load)
            fabric.print(f"[MEM_LOAD] carried memory graph from {_mem_load}: {_injector.store.node_count()} nodes")'''

_MODELLOAD_OLD = '''        state["critics_exploration"] if cfg.checkpoint.resume_from else None,
    )

    # Optimizers'''
_MODELLOAD_NEW = '''        state["critics_exploration"] if cfg.checkpoint.resume_from else None,
    )

    # (C) transfer: carry ONLY the trained world-model weights into a FRESH run (new world,
    # fresh optimizer/ratio so it trains normally). Keeps the latent space consistent with a
    # carried memory graph, without sheeprl's checkpoint-resume suppressing world-B training.
    _model_load = _mem_os.environ.get("MODEL_LOAD", "")
    if _model_load and os.path.exists(_model_load):
        _wm_state = fabric.load(_model_load)
        _wm_state = _wm_state["world_model"] if "world_model" in _wm_state else _wm_state
        # fabric.save stores unwrapped (clean) keys -> load into the unwrapped module
        unwrap_fabric(world_model).load_state_dict(_wm_state)
        fabric.print(f"[MODEL_LOAD] carried world-model weights from {_model_load}")

    # Optimizers'''

_MEMSAVE_OLD = '''    envs.close()
    if _SCOREBOARD_ON and "board" in _SB:'''
_MEMSAVE_NEW = '''    envs.close()
    # (C) transfer: save this world's memory graph so a later, different world can carry it
    _mem_save = _mem_os.environ.get("MEM_SAVE", "")
    if _mem_save and _MEM_ON and _injector is not None:
        _injector.store.persist(_mem_save)
        fabric.print(f"[MEM_SAVE] saved memory graph ({_injector.store.node_count()} nodes) -> {_mem_save}")
    _model_save = _mem_os.environ.get("MODEL_SAVE", "")
    if _model_save:
        fabric.save(_model_save, {"world_model": world_model.state_dict()})
        fabric.print(f"[MODEL_SAVE] saved world-model weights -> {_model_save}")
    _drive_save = _mem_os.environ.get("DRIVE_SAVE", "")
    if _drive_save and _DRIVE_LP and "drive" in _LP:
        _LP["drive"].persist(_drive_save)
        fabric.print(f"[DRIVE_SAVE] saved drive memory ({_LP['drive'].node_count()} nodes) -> {_drive_save}")
    if _SCOREBOARD_ON and "board" in _SB:'''

_ABLDUMP_OLD = '''        fabric.print(report(_SB["board"].out_dir))
    # task test zero-shot'''
_ABLDUMP_NEW = '''        fabric.print(report(_SB["board"].out_dir))
    if _MEM_ABLATE and _ABL["mem"]:
        import json as _json

        _md = _mem_os.environ.get("SCOREBOARD_DIR", os.path.join(log_dir, "scoreboard"))
        os.makedirs(_md, exist_ok=True)
        _m = sum(_ABL["mem"]) / len(_ABL["mem"])
        _n = sum(_ABL["nomem"]) / len(_ABL["nomem"])
        _arm = _mem_os.environ.get("ARM", "run")
        _gain_pct = (100.0 * (_n - _m) / _n) if _n else 0.0
        with open(os.path.join(_md, f"ablation_{_arm}.json"), "w") as _fh:
            _json.dump(
                {"arm": _arm, "samples": len(_ABL["mem"]), "obs_loss_mem": _m,
                 "obs_loss_nomem": _n, "mem_gain": _n - _m, "mem_gain_pct": _gain_pct},
                _fh, indent=2,
            )
        fabric.print(
            f"[ABLATE] obs_loss mem={_m:.3f} nomem={_n:.3f} "
            f"gain={_n - _m:+.3f} ({_gain_pct:+.1f}%) — positive = memory lowers prediction error"
        )
    # task test zero-shot'''

_PROBE_EDITS = [
    (_ABL_FLAG_OLD, _ABL_FLAG_NEW),
    (_ABL_HELPER_OLD, _ABL_HELPER_NEW),
    (_ABL_RUN_OLD, _ABL_RUN_NEW),
    (_MEMLOAD_OLD, _MEMLOAD_NEW),
    (_MODELLOAD_OLD, _MODELLOAD_NEW),
    (_MEMSAVE_OLD, _MEMSAVE_NEW),
    (_ABLDUMP_OLD, _ABLDUMP_NEW),
]


def _sheeprl_dir() -> str:
    spec = importlib.util.find_spec("sheeprl")
    return os.path.dirname(spec.origin)


def patch_memory_aug(path: str, import_anchor: str) -> None:
    with open(path) as fh:
        src = fh.read()
    if "MemoryInjector" in src:
        print(f"  [mem-aug] already patched: {os.path.basename(path)}")
        return
    src = src.replace(import_anchor, import_anchor + _IMPORT_BLOCK, 1)
    for old, new in _MEM_EDITS:
        if old in src and new not in src:
            src = src.replace(old, new, 1)
    with open(path, "w") as fh:
        fh.write(src)
    print(f"  [mem-aug] patched: {os.path.basename(path)}")


def patch_drive_swap(path: str) -> None:
    """Apply the learning-progress drive swap to p2e_dv3_exploration.py.

    Requires the memory-aug phase to have run first (shares the _mem_os/_mem_sys
    import block that _DRIVE_FLAG_NEW appends to).
    """
    with open(path) as fh:
        src = fh.read()
    if "_DRIVE_LP" in src:
        print(f"  [drive-swap] already patched: {os.path.basename(path)}")
        return
    for old, new in _DRIVE_EDITS:
        if old in src and new not in src:
            src = src.replace(old, new, 1)
        else:
            raise RuntimeError(
                f"drive-swap anchor not found in {os.path.basename(path)} (sheeprl version drift?):\n{old[:80]}..."
            )
    with open(path, "w") as fh:
        fh.write(src)
    print(f"  [drive-swap] patched: {os.path.basename(path)}")


def patch_scoreboard(path: str) -> None:
    """Wire the TV noise-trap scoreboard into p2e_dv3_exploration.py.

    Requires the drive-swap phase first (appends after its `_LP` holder line).
    """
    with open(path) as fh:
        src = fh.read()
    if "_SCOREBOARD_ON" in src:
        print(f"  [scoreboard] already patched: {os.path.basename(path)}")
        return
    for old, new in _SCOREBOARD_EDITS:
        if old in src and new not in src:
            src = src.replace(old, new, 1)
        else:
            raise RuntimeError(
                f"scoreboard anchor not found in {os.path.basename(path)} (version drift?):\n{old[:80]}..."
            )
    with open(path, "w") as fh:
        fh.write(src)
    print(f"  [scoreboard] patched: {os.path.basename(path)}")


def patch_probes(path: str) -> None:
    """Wire (B) memory ablation + (C) cross-world transfer into p2e_dv3_exploration.py.

    Requires the scoreboard phase first (shares its `_SB` holder / final block anchors).
    """
    with open(path) as fh:
        src = fh.read()
    if "_MEM_ABLATE" in src:
        print(f"  [probes] already patched: {os.path.basename(path)}")
        return
    for old, new in _PROBE_EDITS:
        if old in src and new not in src:
            src = src.replace(old, new, 1)
        else:
            raise RuntimeError(
                f"probe anchor not found in {os.path.basename(path)} (version drift?):\n{old[:80]}..."
            )
    with open(path, "w") as fh:
        fh.write(src)
    print(f"  [probes] patched: {os.path.basename(path)}")


def main() -> None:
    base = _sheeprl_dir()
    dv3 = os.path.join(base, "algos", "dreamer_v3", "dreamer_v3.py")
    p2e = os.path.join(base, "algos", "p2e_dv3", "p2e_dv3_exploration.py")
    print("Applying integration patch to sheeprl at", base)
    print("Phase 1: memory-augmentation")
    patch_memory_aug(dv3, _IMPORT_ANCHORS["dreamer_v3.py"])
    patch_memory_aug(p2e, _IMPORT_ANCHORS["p2e_dv3_exploration.py"])
    print("Phase 2: learning-progress drive swap (p2e only)")
    patch_drive_swap(p2e)
    print("Phase 3: TV noise-trap scoreboard (p2e only)")
    patch_scoreboard(p2e)
    print("Phase 4: validation probes — memory ablation (B) + cross-world transfer (C)")
    patch_probes(p2e)
    print("Done. Arms: research/p2e/run_arms.sh ; transfer: research/p2e/transfer_test.sh ;"
          " report: research/p2e/monitor/scoreboard.py logs/scoreboard.")


if __name__ == "__main__":
    main()
