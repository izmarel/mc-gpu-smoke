# Artificial Consciousness MVP — Project Context

> **FOR FUTURE SESSIONS:** Read this file first, then TODO.md. This file explains WHAT we're building and WHY. TODO.md explains HOW — the actionable steps, subtasks, and research questions. Do NOT read the Obsidian source files without explicit user permission (see File Access Rule at bottom).

---

## CURRENT STATE — READ THIS FIRST (updated 2026-06-28)

### What is actually built and tested
**POC 1 (the memory layer) only. Nothing else exists yet — no brain, no sensors, no agent.**

- `src/memory/` — vector+graph store, retrieval, derived signals, auto-persist, prune. 75 tests passing.
- Backend: **FAISS (HNSW) for vectors + NetworkX for the graph + pickle for disk.** Atomic snapshots (temp file + `os.replace`), crash-safe.
- Derived signals (confidence / novelty / learning-progress) are simple transparent formulas *by design* — not neural nets.
- Prune cycle: **only the "delete boring nodes" half is built.** Skill-compression (the half that prevents amnesia) is NOT built and is genuinely unsolved.

### Real measured performance (`tests/test_scale.py`, M-series Mac)
- Search at 1,000,000 nodes: ~0.36 ms/query, recall ~96%. **Search is never the bottleneck** (fine well past 1M).
- Insert, one node at a time, DEGRADES: ~9,300/s at 10k → ~800/s at 100k. **This is the real ceiling.**
- Disk save/load at 1M: ~4 s each (only happens during sleep).
- Practical ceiling ≈ 1M nodes before insert can't hold 30fps. Storing every frame at 30fps: the default 50k sleep threshold = **~28 min awake**; raising it to ~1M ≈ **a few hours**. 16h+ awake requires NOT storing every frame (surprise-gated storage) — but that is **premature** until a real agent produces real data to tune against.

### Where the first ("hallucinator") LLM was wrong — DO NOT REPEAT
1. **Survival / battery / metabolic drive is REJECTED.** It came from the hallucinator's MVP plan, not this project's philosophy. The entity has ONE driver: building a better model of reality (curiosity). **Scarcity breeds malice** — a creature forced to compete for survival learns adversarial behavior. There is no battery, no food, no death.
2. **The reward is LEARNING PROGRESS, not raw surprise.** Raw prediction error is the "noisy-TV" trap: random noise has permanently high error and would hypnotize a naive agent forever. The drive must reward prediction error *trending DOWN over time* (actually learning), so noise → no progress → boredom → it leaves.
3. **The "master architecture" (Amygdala / Spinal Cord / Virtual Womb…) is a concept collage** — real terms, fake wiring. Ignore it as architecture; mine it only for standard components hiding under dramatic names.
4. **Database wrong-turn (this project's own, now fixed):** LadybugDB was chosen, then benchmarked and REJECTED (too slow past 10k). Wherever older notes say LadybugDB, the truth is **FAISS + NetworkX**.

### The single drive, stated cleanly
**Build a better model of reality.** Everything — exploration, memory, the eventual LLM — serves that. Destruction self-punishes because it destroys future learning. No survival pressure needed or wanted.

### What needs to be built NEXT
The **Curiosity Grid (POC 0)** — the simplest possible proof of the core bet, BEFORE any more memory work. Full task breakdown in `TODO.md` → "POC 0". One line: a dot on a grid driven purely by learning-progress curiosity; prove (1) it explores then goes calm once it understands the grid, and (2) a planted **noise square** does NOT trap it. This is the **lie-detector** for the whole project. Memory-safety work (skill-compression, surprise-gate, trickle-prune) is DEFERRED until POC 0 produces real data.

## Origin

User had a long philosophical conversation with an LLM about building artificial consciousness. The first LLM produced a hallucinated "master architecture" — real terms (FEP, Mamba, KL divergence) with fake plumbing between them. A second, more honest conversation (4507 lines) pushed back on every bullshit tendency and iteratively arrived at a cleaner architecture through honest give-and-take. The user explicitly rejects LLM sycophancy, pattern-matching onto named frameworks, and dressing up limitations as design choices.

Source files (Obsidian notes — **DO NOT read without explicit user permission**):
- `~/Documents/Obsidian/Notes/Artificial Consciousness/MVP v1/Architecture finalized, big convo.md` — the 4507-line conversation
- `~/Documents/Obsidian/Notes/Artificial Consciousness/MVP v1/Architecture.md` — the hallucinated "V1.0 master architecture" (criticized as a concept collage)
- `~/Documents/Obsidian/Notes/Artificial Consciousness/MVP v1/MVP plan.md` — MVP plan + iteration path (partially from the hallucinator)
- `~/Documents/Obsidian/Notes/Artificial Consciousness/MVP v1/Visualizing MVP.md` — telemetry/debugging visualization plan
- `~/Documents/Obsidian/Notes/Artificial Consciousness/AI on Artificial Consciousness.md` — early Gemini conversation
- `~/Documents/Obsidian/Notes/Artificial Consciousness/Interesting debate with AI about consciousness and building an observer-driven AI.md` — another early conversation

---

## The Drive — Single Foundational Principle

**The system wants to build a better model of reality.**

Everything derives from this. Chosen because:
- Provides direction for all behavior
- Naturally constrains destructive behavior — destroying things or harming agents reduces future learning opportunities, so the drive punishes itself
- Unifies three earlier observations: reduce uncertainty + maintain ability to learn + model other agents accurately

---

## The Signal

A snapshot at any given moment across ALL peripherals — both input and output.

- **X axis:** spread across peripherals (cam, mic, temp, accel, motor feedback/proprioception, language input)
- **Y axis:** agitation level of each peripheral
- Output peripherals are part of the signal. When the system acts, the action and its consequences show up through normal sensory channels. Camera sees the arm move. Accelerometer feels it. That IS the consequence feedback. No separate consequence sensor needed. Only motor-specific feedback is proprioception (joint angles, motor positions) — just another peripheral.
- **Tick rate:** runs at vision refresh rate (30-60fps) as vision is top of the sensory hierarchy. Slower peripherals return stale data until they update. Faster peripherals get averaged or downsampled.

### Signal Encoding

Each peripheral has a frozen pretrained encoder that outputs a vector. Each vector goes through a small **trained** projection layer to a common dimensionality (256). All get concatenated into one fixed-size state vector. Adding a new peripheral = new encoder + new projection layer. Modular.

**Projection layers are trained properly** (LLaVA-style linear layer on simulation data, hours on single GPU). NOT PCA, NOT truncation — both destroy semantic information and were explicitly rejected by the user.

**Open question (TODO 2.5a):** Can native encoder dimensions be used for memory storage (LadybugDB) without projection, deferring projection training to POC 4 (world model integration)? This would let POC 1-3 work without any projection training.

### POC Sensors — Use The Computer Itself

No need to buy hardware. The computer IS a sensor platform:

| Peripheral | Source | Encoder | Python lib |
|---|---|---|---|
| Vision | Screen capture or webcam | CLIP → 768-dim | mss / Pillow / opencv |
| Audio | Microphone | Whisper → vector | sounddevice |
| "Proprioception" | Mouse X/Y, keyboard state | MLP → 256 | pynput (read state) |
| "Motor output" | Mouse moves, clicks, keys | This is the ACTION space | pyautogui (write actions) |

The "world" is the desktop. The "actions" are mouse + keyboard. Real multi-peripheral input, real consequences (clicking shit does things), no hardware cost.

---

## Memory — Unified Vector Graph

One single data structure. Not separate systems.

### Infrastructure: FAISS + NetworkX (BUILT) — LadybugDB rejected

> **CORRECTION:** Earlier drafts of this doc named LadybugDB. That was benchmarked and **rejected** (vector search too slow past ~10k nodes — 4.4ms at 10k, scaling linearly, blows the 16-33ms tick budget). The actual, built backend is FAISS + NetworkX. Treat any remaining "LadybugDB" mention in old notes as historical.

**What's implemented (`src/memory/faiss_networkx_store.py`):**
- **FAISS `IndexHNSWFlat`** (approximate nearest-neighbour, sub-ms at 1M) for the vector index. HNSW has no in-place delete, so deletion is a tombstone + periodic rebuild (handled internally).
- **NetworkX `MultiDiGraph`** for the typed-edge graph (ACTION / RESULT / CONTEXT) and traversal.
- **pickle** for persistence — graph + id maps written atomically (temp file + `os.replace`); the FAISS index is a rebuildable cache reconstructed on load (one file, no two-file skew).
- A dependency-free `InMemoryStore` reference backend (`src/memory/storage.py`) defines the behavioural contract every backend must match.

No ACID and no built-in crash safety — that is why auto-persist (TODO 1.10, built) and the prune/sleep cycle (TODO 1.11, half-built) exist.

**Research needed (TODO 1.1a, 1.1b):** Verify LadybugDB's Python API for vector similarity search (Cypher syntax, cosine vs L2, speed at scale) and graph traversal (typed edges, multi-hop traversals, persistence, CRUD).

### Node Structure
```
NODE:
  - state vector (projected dims, e.g., 768 or 1792 depending on POC stage)
  - prediction error (float — how surprising was this moment)
  - timestamp (unix — when this happened)
  - text metadata (string, optional — what internal monologue was active)

EDGES FROM THIS NODE:
  - ACTION edge → what action was taken
  - RESULT edge → next state node (what happened after)
  - CONTEXT edges → other memory nodes that were retrieved/active
    when this happened (linking this experience to similar pasts)
```

This creates natural chains: state → action → result/next state → action → result/next state. These chains are **episodes of experience**. Context edges create cross-links between different episodes.

### Retrieval

State vector queries LadybugDB's vector index. Returns most similar past state nodes in milliseconds. Then edges are followed to get actions taken, results that followed, and context that was active.

**Open questions on retrieval:**
- How many nodes to retrieve (5? 50?)
- How many hops deep to follow edges (one hop to get action and result, or deeper?)
- Retrieved info is combined with current state vector using **position tags** — small learned vectors that mark each chunk as "current state" or "retrieved memory N". Standard approach in retrieval-augmented systems.

### Confidence From Retrieval

Prediction confidence is NOT a separate module. Falls out of retrieval results:
- Many similar past states + consistent low prediction error = **HIGH confidence**
- Few similar states or inconsistent outcomes = **LOW confidence**
- No similar states at all = **VERY LOW confidence** (completely novel territory)

### Novelty From Retrieval

- Few/no similar nodes found = HIGH novelty
- Similar nodes found but far away in vector space = MODERATE novelty
- Many close similar nodes = LOW novelty

(POC 4+ when world model exists, novelty switches to world model prediction error trend — more accurate, already computed, zero extra cost.)

### Memory Growth and Pruning

Graph grows with every experience. During rest/sleep:
- **COMPRESS** frequently-used low-error chains into skill nodes (well-learned skills). Mechanics still undefined — could be centroid averaging, prototype+variance, or edge strengthening with intermediate deletion.
- **PRUNE** infrequent low-error nodes (boring, already predicted well, not part of active skill)
- **KEEP** high prediction error nodes (surprising, informative regardless of frequency)
- **KEEP** highly connected nodes (important junctions in experience)

**Skill preservation problem:** if you opened a door a million times with low prediction error, deleting those nodes means forgetting how to open doors. So low-error nodes that are part of heavily-used chains should be COMPRESSED rather than deleted. Frequent + low error = compress into strong skill paths. Infrequent + low error = safe to prune. High error = keep regardless.

---

## Output — The Spine

Output peripherals (robotic arm, servos, voice/language output, etc.) connected through the world model's action decoders. Different output heads for different peripherals.

### Motor Output
World model outputs an action vector. A motor decoder (linear projection) maps this to specific joint commands. DreamerV3 handles this for moderate complexity — robotic arms, quadrupeds, up to 20-30 joints. Prototype: 6-7 DOF arm with gripper.

For POC: mouse/keyboard output (pyautogui) instead of robot arm.

### Language Output
Two directions, both with existing proven approaches:

**Signal → LLM (for internal monologue and language reasoning):**
Compressed state vector → trained projection network → LLM embedding space. LLM receives it as if it were input tokens. This is the **LLaVA approach** applied to state vectors instead of images. Projection needs training on paired data (state vectors + text descriptions, generated during simulation).

**LLM → Signal (for language feedback and action suggestions):**
- For motor action suggestions: **RT-2 approach**. LLM outputs discretized action tokens. Parsed into motor commands. Crude but proven.
- For internal monologue feedback: LLM's text output → text encoder → vector → injected into signal as language_input peripheral on next tick.

**Open question (TODO 5.2):** Can we skip the trained projection entirely for POC and just format state as text? "I'm at position (4,7), battery 73%, food at (8,2)" → send directly to LLM. Cruder but zero training data needed. Try text-first, train projection only if text approach is insufficient.

---

## World Model — DreamerV3-Style

The core mechanism that predicts what happens next given the current state and a proposed action.

### Why DreamerV3
Only existing open-source system that learns a world model from raw sensory input and plans inside that model. Proven on simulated robots. Runs on a single GPU. The hallucinator's exotic components (Critic/Amygdala, Spinal Cord, etc.) are mostly standard parts of this kind of system with dramatic names.

### SheepRL — The Implementation

**SheepRL** is a PyTorch RL framework (`pip install sheeprl`) that includes DreamerV3 AND **Plan2Explore**.

**Plan2Explore** is critically relevant — it's a Dreamer variant designed for **curiosity-driven exploration without external reward**. The agent explores purely based on how much it can improve its world model. That IS the drive: "build a better model of reality." Already implemented, not theoretical.

Config: `sheeprl/configs/algo/p2e_dv3.yaml` (inherits dreamer_v3, adds ensembles).

Benchmarks: DreamerV3 on Crafter in 1 day 3 hours on a single V100. Atari in 14 hours on a single 3080. Consumer GPU scale, not datacenter.

### DreamerV3 Memory Integration (investigated and verified — TODO 4.1)

SheepRL source code cloned and inspected at `/tmp/sheeprl_inspect/`. Key findings (verified against actual code, not training-data guesswork):

**Actual files:**
- `sheeprl/algos/dreamer_v3/agent.py` — RSSM, MLPEncoder, CNNEncoder, RecurrentModel, decoders, actor, critic
- `sheeprl/algos/dreamer_v3/dreamer_v3.py` — main algorithm, `train()` function, player loop
- `sheeprl/algos/dreamer_v3/loss.py` — `reconstruction_loss()`
- Configs are hydra YAML in `sheeprl/configs/algo/dreamer_v3.yaml`

**Injection point:** `dreamer_v3.py:113` — `embedded_obs = world_model.encoder(batch_obs)`. The encoder takes a dict of tensors. Add a "memory" key to the dict with retrieved memory vectors and the MLPEncoder handles it (concatenates all values).

**What stays untouched:** RSSM (posterior gets embedded_obs, doesn't care what went into encoder), decoder, actor, critic, loss function, imagination loop (uses RSSM prior, no observation input).

**What needs changing:** encoder input dim (trivial — bump constructor arg), replay buffer to store memories (medium — TODO 4.2 research needed), player loop to retrieve+inject memories (medium), config YAML (trivial).

**VERDICT: Tractable with LLM assistance. No ML expert needed.** It's plumbing, not research. Hard part is replay buffer sequence alignment.

### What DreamerV3 Doesn't Have (needs to be added)
- Memory graph integration (it has no persistent structured memory)
- Curiosity/boredom drive — **BUT Plan2Explore variant already solves this**
- Confidence-gated action vs planning (it always plans the same way)
- Multiple input/output modalities in the way described
- Language peripheral connection

### How It Updates
Fixed-size neural network. Does NOT update in real-time on every experience. Experiences go into a **replay buffer**. Periodically, in batches, the world model trains on sampled experiences. System is always acting on a slightly stale model.

**Sleep/rest trigger:** when replay buffer hits capacity, stop collecting new experience and process: train world model on buffer, consolidate important experiences into memory graph, prune/compress graph, empty buffer, resume.

### Simulation / Planning
When confidence is low, world model simulates multiple possible actions by running predicted trajectories forward internally. Evaluates which trajectory leads to best predicted outcome and picks that action.

Confidence Gate routing:
- confidence > HIGH threshold → FAST PATH: act directly, single prediction, no planning
- confidence > LOW threshold → PLAN PATH: simulate multiple trajectories, evaluate, pick best
- confidence < LOW threshold → LANGUAGE PATH: translate state to LLM, reason in language, get candidates, evaluate via simulation, pick best. System executes safe default action (wait/retreat) while LLM processes.

Thresholds: initially hardcoded, tuned experimentally. Later potentially learned from experience.

---

## Drive Mechanics — Three Signals From Prediction Error

All based on prediction error, measured at different time windows. **NO SEPARATE BOREDOM MODULE** — it was killed as unnecessary overhead.

### Boredom / Restlessness Signal
Chronic low prediction error over sustained period = everything is predictable = nothing to learn. Signal increases, creates internal agitation that pushes system to seek novelty.

(POC 1-3: derived from memory retrieval — few/no similar memories = novel. POC 4+: derived from world model prediction error — high = novel. Already computed, zero extra cost.)

### Learning Reward
Prediction error TRENDING DOWNWARD over recent experience with a particular kind of thing. System is actively getting better at predicting this. This is the actual reward signal. Positive when system is making progress.

**Solves the noisy TV problem:** Random noise has permanently high prediction error that never trends downward. Learning reward for watching noise is zero. System gets bored and moves on. Only things where prediction error actually decreases produce sustained reward.

### Prediction Error Itself
Used for:
- Surprise detection (something unexpected just happened)
- Memory importance tagging (high prediction error moments are more valuable to store and keep)
- Informing confidence through retrieval

### The Balance / Loop
1. Low novelty (familiar states) → seek something new
2. High novelty → low confidence → plan carefully
3. Engage with novel thing
4. Learning progress positive → stay engaged
5. Learning progress flat → disengage, seek new thing
6. Mastery reached (low novelty again) → back to 1

### Will (special case)
Learning progress positive but slow. Current prediction error high (hard task). System continues because downward trend in error = reward outweighs discomfort. No special mechanism — world model learns from memory that persistence in positive-trend states eventually leads to mastery.

---

## What Derives From The Drive

### Motivation
Uncertainty gap IS the motivation. Bigger gaps pull harder. No separate motivation system.

### Learning
Experiential first — act, observe, update. Language reasoning as fallback when stuck (try → fail → think in words → replan → try again → succeed → store so language isn't needed next time).

### Agency
System must model itself as a causal actor because its own actions are part of reality. Self-modeling is demanded by the drive, not bolted on.

### Will
Staying in high-agitation states because projected learning reward outweighs current discomfort. Requires planning and impulse suppression (ability to NOT immediately react).

### Planning
Pattern match current state against memory (vector similarity retrieval). Follow edges to see past actions and outcomes. Simulate candidate actions through world model. Pick best trajectory. If simulation unhelpful, escalate to language reasoning.

### Internal Monologue
Language output peripheral routed back into the signal as input. The system talks to itself. Used for abstract reasoning, planning, and as the mechanism through which the self-model is expressed and examined.

### Self-Model
NOT a separate module. Emerges from memory (the system's own experiences stored in the graph) plus internal monologue (language-based reasoning about those experiences). Core beliefs form as compressed, heavily-reinforced patterns in the graph. Self-reflection is the system querying its own experiential history through language and evaluating whether stored patterns still match current evidence.

The self-model has inertia — it resists change. Early experiences weigh more heavily. Later rewiring is possible but expensive. Some beliefs become deeply entrenched and changing them has diminishing returns.

### Morality
Learned through consequences, not programmed. Harmful actions empirically degrade the system's ability to build a better model (agents become adversarial, environment degrades, unpredictability increases). Training environment must include other agents and realistic consequences.

### Rest / Sleep
Triggered when replay buffer hits capacity. System pauses. Consolidate memories, compress skills, prune, train world model, resume.

### Evolving
Behavior naturally shifts as model improves. Early: broad exploration (everything is novel). Later: narrow deep exploration (strong model, incremental improvements). Meta-level evolution of the learning process itself is still undefined.

---

## Language Pathway

Runs asynchronously alongside the main tick loop. Does NOT block the main loop. ~2-5hz.

### Mode 1: Internal Monologue (continuous background)
Every ~200-500ms:
1. World model's current internal state captured
2. State vector projected into LLM embedding space (trained projection, LLaVA-style) OR formatted as text (TODO 5.2)
3. LLM generates brief internal narrative
4. Generated text stored as metadata on current memory node
5. Text encoded through text encoder → vector → injected into language_input slot in signal
6. Next tick, world model sees this as part of the signal

### Mode 2: Reasoning Fallback (triggered by very low confidence)
1. Current state + retrieved memories + their text metadata formatted into richer prompt
2. LLM reasons more extensively about the situation
3. LLM outputs candidate actions (RT-2-style discretized tokens, or text parsed to action vectors)
4. Candidate actions sent to world model for evaluation via simulation
5. Best candidate selected and executed
6. If successful: whole experience stored in memory graph so next time language fallback isn't needed

While language pathway processes in Mode 2, main tick loop continues with safe default action (wait, retreat, hold position).

---

## Existing Technology Stack

| Component | Technology | Status |
|---|---|---|
| Memory (vectors + graph + persistence) | **FAISS (HNSW) + NetworkX + pickle** | BUILT & TESTED — `src/memory/`, 75 tests, scale-verified to 1M. (LadybugDB benchmarked and rejected.) |
| Sensory encoders | CLIP (vision, open_clip_torch ViT-B/32), Whisper (audio, tiny), MLPs (sensors) | Proven, available |
| Projection layers | Trained LLaVA-style linear layers (NOT PCA/truncation) | Standard practice, needs training data |
| World model + planning | DreamerV3 via SheepRL (pip install sheeprl) | Open source, proven, code inspected |
| Curiosity/novelty drive | Plan2Explore (SheepRL) — world model improvement as reward | Proven, already implemented |
| Signal → LLM translation | LLaVA-style projection OR text formatting (TODO 5.2) | Proven approach, needs adaptation |
| LLM → action | RT-2-style tokens OR text parsing (TODO 5.5) | Proven approach, text parsing for POC |
| Small fast LLM | Local via ollama or llama.cpp | Available, good on Apple Silicon |
| Screen capture | mss | Proven, cross-platform |
| Audio capture | sounddevice | Proven, simple |
| Mouse/keyboard | pynput (read), pyautogui (write) | Proven |
| Simulated training env | MuJoCo, Crafter (via SheepRL) | Proven, available |

---

## POC Plan (summary — see TODO.md for full subtasks)

1. **POC 1: Memory graph + retrieval** — LadybugDB, fake data, test storage/retrieval/confidence/novelty/learning-progress. Days-weeks. HIGH feasibility.
2. **POC 2: Signal encoding** — CLIP/Whisper/MLPs → trained projections → unified state vector. Days. HIGH.
3. **POC 2.5: Run Plan2Explore** — pip install sheeprl, run, observe curiosity-driven exploration. Hours. HIGH.
4. **POC 3: Decision loop** — sense → remember → decide (confidence-gated) → act → learn, with LLM fallback. Weeks-months. MEDIUM.
5. **POC 4: DreamerV3 + memory** — inject memory into DreamerV3 encoder (verified tractable, no ML expert needed). Months. MEDIUM with LLM help.
6. **POC 5: Language pathway** — internal monologue, LLM reasoning, self-model emergence. Months. MEDIUM with LLM help (text-first approach may skip projection training).

---

## Genuinely Unsolved / Experimental

- Skill compression mechanics in the vector graph (what does a compressed skill node look like?)
- Full integrated loop at real-time tick speed (encode → retrieve → predict → decide → act within 16-33ms)
- Whether language reasoning actually helps in practice or is just noise
- Whether confidence thresholds stabilize through learning
- Meta-learning / evolution of learning process
- Training environment design for moral learning
- Sim2real transfer with all components running
- Whether any of this produces consciousness or just a sophisticated robot

---

## Hardware

MacBook M5 Pro 48GB. Unified memory is advantage — GPU can access all 48GB.
- POC 1-3: definitely viable (LadybugDB, CLIP, Whisper all run fine on Mac)
- Plan2Explore: probably works but slower than NVIDIA (maybe 2-3x slower training). Annoying but not a blocker.
- LLM inference: actually good on Apple Silicon (llama.cpp, MLX optimized for it). 7-13B parameter model runs comfortably in 48GB.

---

## File Access Rule

User explicitly instructed: **read only files that are explicitly named/pointed to. No filesystem scanning.** Do not use broad `search_files` across directories. If you need to find something, ask the user where it is.

---

## User Preferences

- Communicates with heavy profanity, confrontational and direct. Expects uncensored, vulgar, no-bullshit responses.
- Cares deeply about token efficiency. Use Claude Code CLI for subagents/delegation, NOT Hermes subagents (which burn too many tokens).
- Hates sycophancy, "you're right", "I'm sorry", and AI-isms. Be direct.
- Wants real numbers, file paths, and verifiable specifics — not hand-waving.
- PCA/truncation for projections was explicitly rejected. Train properly or don't bother.
- The hallucinated "master architecture" was criticized as a concept collage — real terms with fake wiring. Do not repeat that pattern.