# SheepRL SequentialReplayBuffer — Internals (TODO 4.2)

Research target: how the DreamerV3 replay buffer stores/samples data, and exactly what to
change to add a `"memories"` key. This is plumbing research, not ML research.

**Source:** `dprato87/sheepRL` @ `33b6366`, cloned to `/tmp/sheeprl_research`.
Files inspected:
- `sheeprl/data/buffers.py` (1180 lines)
- `sheeprl/algos/dreamer_v3/dreamer_v3.py` (780 lines)
- `sheeprl/algos/dreamer_v3/agent.py` (1236 lines)
- `sheeprl/models/models.py` (MultiEncoder/MLPEncoder)
- `sheeprl/algos/dreamer_v3/utils.py` (`prepare_obs`)
- `sheeprl/configs/algo/dreamer_v3.yaml`, `sheeprl/configs/exp/dreamer_v3.yaml`

**TL;DR:** The buffer schema is **fully flexible** — it is whatever dict of numpy arrays you
hand to `rb.add()` on the first call. Every array shares the first two dims `[seq_len, n_envs]`;
the rest is free. Adding a `"memories"` key is **static and easy**: put a `memories` array into
`step_data` (and `reset_data`) at collection time, add `"memories"` to `mlp_keys.encoder`, keep it
**out** of `mlp_keys.decoder`, and register a matching gym observation space. No buffer code changes
required.

---

## 1. How the buffer stores data

### The actual buffer used by DreamerV3

`dreamer_v3.py:479-485` — the buffer is an `EnvIndependentReplayBuffer` wrapping one
`SequentialReplayBuffer` **per environment**:

```python
rb = EnvIndependentReplayBuffer(
    buffer_size,
    n_envs=cfg.env.num_envs,
    memmap=cfg.buffer.memmap,
    memmap_dir=os.path.join(log_dir, "memmap_buffer", f"rank_{fabric.global_rank}"),
    buffer_cls=SequentialReplayBuffer,
)
```

`EnvIndependentReplayBuffer.__init__` (`buffers.py:575-586`) builds a list of `n_envs` independent
`SequentialReplayBuffer`s, each with `n_envs=1`. Add/sample fan out to / concatenate across these.

### Internal representation: dict of numpy arrays (or memmaps)

`ReplayBuffer.__init__` (`buffers.py:60`): `self._buf: Dict[str, np.ndarray | MemmapArray] = {}`.
`SequentialReplayBuffer` inherits this storage unchanged; it only overrides `sample()`/`_get_samples()`.

- Storage is a **Python dict** mapping `str -> np.ndarray` (or `MemmapArray` if `cfg.buffer.memmap=True`).
- Each array has shape **`(buffer_size, n_envs, *feature_dims)`** — see allocation at `buffers.py:214`:
  ```python
  self.buffer[k] = np.empty(shape=(self._buffer_size, self._n_envs, *v.shape[2:]), dtype=v.dtype)
  ```
  (For the per-env sub-buffers `n_envs == 1`.)
- Data added must be shaped `[sequence_length, n_envs, ...]` (docstring `buffers.py:147-148`).
- **Not tensors.** Tensors only appear at sample time via `sample_tensors()` / `get_tensor()`
  (`buffers.py:1158-1180`), which wraps `torch.as_tensor` / `torch.from_numpy`.
- `_pos` is the circular write head; `_full` flips once it wraps (`buffers.py:219-221`). When full,
  oldest data is overwritten.

### What keys exist (DreamerV3)

The keys are **not hardcoded in the buffer** — they are exactly the keys present in the `step_data`
dict the algorithm writes. For DreamerV3 (`dreamer_v3.py:539-546, 586, 628-637`):

| Key | Set at | Shape per step (`[1, n_envs, ...]`) | Notes |
|---|---|---|---|
| `<cnn_keys.encoder>` e.g. `rgb` | `:542`, `:629` | `[1, n_envs, C, H, W]` | uint8 images |
| `<mlp_keys.encoder>` e.g. `state` | `:542`, `:629` | `[1, n_envs, D]` | vector obs |
| `actions` | `:586` | `[1, n_envs, sum(actions_dim)]` | |
| `rewards` | `:543`, `:637` | `[1, n_envs, 1]` | |
| `terminated` | `:545`, `:635` | `[1, n_envs, 1]` | |
| `truncated` | `:544`, `:636` | `[1, n_envs, 1]` | |
| `is_first` | `:546`, `:594` | `[1, n_envs, 1]` | |

`obs_keys = cfg.algo.cnn_keys.encoder + cfg.algo.mlp_keys.encoder` (`dreamer_v3.py:433`) and is passed
to the buffer as the set of keys eligible for `next_*` sampling.

---

## 2. How sequence sampling works

The training pull is at `dreamer_v3.py:664-671`:

```python
local_data = rb.sample_tensors(
    cfg.algo.per_rank_batch_size,
    sequence_length=cfg.algo.per_rank_sequence_length,
    n_samples=per_rank_gradient_steps,
    dtype=None,
    device=fabric.device,
    from_numpy=cfg.buffer.from_numpy,
)
```

> **Note on the "default 15":** in this repo `per_rank_sequence_length` is a **required** field
> (`dreamer_v3.yaml:19` is `???`). The real experiment configs set it to **64**
> (`configs/exp/dreamer_v3.yaml:14`), matching the DreamerV3 paper's batch length. The "15" in the
> TODO is stale — the sampling logic is identical regardless of the value.

### Path: `EnvIndependentReplayBuffer.sample` → per-env `SequentialReplayBuffer.sample`

`EnvIndependentReplayBuffer.sample` (`buffers.py:656-699`) splits `batch_size` randomly across the
per-env buffers (`np.bincount(...)`, `:684`), samples each, then concatenates the per-buffer results
along `batch_axis` (`:698`). For `SequentialReplayBuffer`, `batch_axis = 2` (`buffers.py:364`), i.e.
sequences are concatenated along the **batch** axis of `[n_samples, seq_len, batch, ...]`.

### Core sequence sampling — `SequentialReplayBuffer.sample` (`buffers.py:395-465`)

1. `batch_dim = batch_size * n_samples` (`:420`).
2. Compute **valid start indices** so a sequence of `sequence_length` never crosses the write head
   `self._pos` (which holds an invalid/in-progress transition):
   - **Full buffer** (`:439-453`): valid range is `[0, pos - seq_len + 1) ∪ [pos, second_range_end)`;
     pick `batch_dim` start indices from it.
   - **Not full** (`:454-456`): `start_idxes = rng.integers(0, pos - seq_len + 1, size=(batch_dim,))`.
3. Build the sequence index grid (`:458-460`):
   ```python
   chunk_length = np.arange(sequence_length).reshape(1, -1)            # (1, seq_len)
   idxes = (start_idxes.reshape(-1, 1) + chunk_length) % self.buffer_size   # (batch_dim, seq_len)
   ```
   So each row is `[start, start+1, ..., start+seq_len-1]` — **consecutive** timesteps (sequentiality),
   wrapping modulo `buffer_size`. Episode boundaries are **ignored** (docstring `:404-405`); `is_first`
   carries reset info into training instead.

### Gather + reshape — `_get_samples` (`buffers.py:467-526`)

- One env index per sequence (each sequence comes from a single env; here per-env buffers have
  `n_envs=1`, so `env_idxes = 0`, `:480-481`).
- `flattened_idxes = flattened_batch_idxes * n_envs + env_idxes` (`:489`) then
  `np.take(reshape(v, (-1, *feat)), flattened_idxes, axis=0)` per key (`:497`).
- Reshape to `(n_samples, batch_size, seq_len, *feat)` then `swapaxes(1, 2)` →
  **`(n_samples, seq_len, batch_size, *feat)`** (`:505-511`).
- If `sample_next_obs=True`, adds `next_<obs_key>` for keys in `obs_keys` (`:514-525`). DreamerV3
  calls with the default `sample_next_obs=False` (it reconstructs the current obs and uses `is_first`),
  so no `next_*` keys here.

### Crucial property: keys stay aligned

`_get_samples` loops `for k, v in self.buffer.items()` and applies the **same `idxes`** to every key
(`:493`). So every key is sampled at the **same timesteps** — a `memories` key added alongside the
observation is automatically sequence-aligned with `actions`, `rewards`, `is_first`, etc. This is the
"replay buffer sequence alignment" concern from TODO 4.1 — and it is handled for free as long as
`memories` is just another key in the same dict.

---

## 3. Exact data structure returned by the sampler

`rb.sample_tensors(...)` returns:

```python
Dict[str, torch.Tensor]   # one entry per stored key
# each tensor shape: [n_samples, sequence_length, batch_size, *feature_dims]
```

(`buffers.py:731-732` docstring; for `SequentialReplayBuffer`.)

Then the train loop iterates the `n_samples` (= gradient steps) axis (`dreamer_v3.py:673, 681`):

```python
for i in range(per_rank_gradient_steps):
    batch = {k: v[i].float() for k, v in local_data.items()}   # -> [seq_len, batch_size, *feat]
    train(..., batch, ...)
```

Inside `train`, the encoder is fed (`dreamer_v3.py:98-99, 113`):

```python
batch_obs = {k: data[k] / 255.0 - 0.5 for k in cfg.algo.cnn_keys.encoder}
batch_obs.update({k: data[k] for k in cfg.algo.mlp_keys.encoder})
...
embedded_obs = world_model.encoder(batch_obs)               # the injection point (TODO 4.1)
```

So per-key tensors arrive as `[sequence_length, batch_size, feature_dim]`.

---

## 4. Does the buffer have a fixed schema? — No, it's flexible

- The buffer has **no declared schema**. On the first `add()` with an empty buffer it allocates one
  `np.empty` array per key found in the data dict (`buffers.py:212-215` / memmap branch `:203-211`).
- Keys are defined **implicitly by the first batch of data**. `obs_keys` only controls which keys get
  `next_*` companions when `sample_next_obs=True`; it does not restrict storage.
- **Constraints (from `add`, `buffers.py:160-221`):**
  1. Data must be a `dict[str, np.ndarray]` (validated only if `validate_args=True`, `:162-192`).
  2. Every array must have ≥2 dims and be **congruent in the first two dims** `[seq_len, n_envs]`
     (`:176-190`). The trailing feature dims are unconstrained and can differ per key.
  3. After the first add, every subsequent add must carry the **same key set** — `add` does
     `self.buffer[k][idxes] = data_to_store[k]` for each key in the *incoming* data (`:217-218`).
     A missing key silently won't be updated; an **extra** key not allocated on the first add will
     `KeyError`. So: decide the key set up front and include `memories` from the very first add.

`EpisodeBuffer` (used by some algos, not DreamerV3 here) additionally **requires** `terminated` and
`truncated` keys (`:926-929`) — DreamerV3 uses `SequentialReplayBuffer`, which has no such requirement.

---

## 5. Adding a `"memories"` key — static vs dynamic

### Static (store at collection time) — RECOMMENDED, easy

Because the schema is flexible, you store the retrieved memory vector as just another observation key.
No changes to `buffers.py` at all. Steps:

1. **Player loop / collection** — when an action is taken, retrieve memories for the current obs and
   put them in `step_data` **before** `rb.add` (`dreamer_v3.py:586-587`):
   ```python
   # after computing the action, before rb.add(step_data, ...)
   memory_vec = memory_store.retrieve(obs)            # -> np.ndarray [num_envs, memory_dim]
   step_data["memories"] = memory_vec[np.newaxis]     # -> [1, num_envs, memory_dim]
   rb.add(step_data, validate_args=cfg.buffer.validate_args)
   ```
2. **Reset data** — the `reset_data` dict written on episode end (`dreamer_v3.py:642-650`) must also
   include `memories`, or you'll hit a key-mismatch on those adds. Add a zero/placeholder:
   ```python
   reset_data["memories"] = np.zeros((1, reset_envs, memory_dim))
   ```
   (Also seed it in the initial `step_data` block at `:539-546` so the very first add defines the key.)
3. **Encoder** — add `"memories"` to `cfg.algo.mlp_keys.encoder`. Then it automatically flows into
   `batch_obs` (`dreamer_v3.py:99`) and through `MLPEncoder`, which concatenates all `mlp_keys`
   (`agent.py:149-151`, `models.py:465-475`). Encoder input dim grows by `memory_dim` automatically
   (`agent.py:998`: `input_dims=[obs_space[k].shape[0] for k in mlp_keys.encoder]`).
4. **Keep it out of the decoder** — set `cfg.algo.mlp_keys.decoder` explicitly to your real obs keys,
   **excluding** `memories`. Otherwise the world model will try to reconstruct memories in
   `reconstruction_loss` (`dreamer_v3.py:156-161, 176`) — wasteful and probably undesired. Validation
   at `dreamer_v3.py:413-427` only requires decoder keys ⊆ encoder keys and a non-empty intersection,
   so encoder-only `memories` is allowed.
5. **Observation space** — `build_agent` reads `obs_space["memories"].shape[0]` (`agent.py:998`).
   Register a matching `gym.spaces.Box(shape=(memory_dim,))` in the env's Dict observation space (via
   an env wrapper) so the encoder is built with the right input dim.

That's it. The sampler returns `memories` aligned with every other key, shaped
`[n_samples, seq_len, batch_size, memory_dim]`, and it lands in `batch_obs["memories"]` automatically.

### Dynamic (retrieve at sample time) — harder, defer

To retrieve memories during training you'd query the memory store using the **sampled** states inside
`train()` (after `local_data` is pulled, `dreamer_v3.py:664`). Problems:
- You need the raw state to query with; the sampled obs is available, but the retrieval call would run
  on `seq_len × batch_size` states every gradient step — expensive and breaks the clean
  numpy-in-buffer flow.
- The memory store keeps changing during training, so retrieved memories would be inconsistent with
  what was seen at collection time (which can be a feature or a bug).
- Sequence alignment must be redone manually per `[seq_len, batch]` element.

**Recommendation (matches TODO 4.2): start static.** It requires zero buffer modifications and the
buffer's same-`idxes`-for-all-keys behavior gives free sequence alignment. Upgrade to dynamic only if
stale memories prove to be a real problem.

---

## 6. Where the buffer is populated during training

All in `dreamer_v3.py`, inside the `main()` collection loop (`for iter_num in range(...)`, `:550`):

- **Initial seed of `step_data`** — `:539-547`: obs keys, `rewards`, `truncated`, `terminated`,
  `is_first`, then `player.init_states()`.
- **Per-step add** — `:586-587`:
  ```python
  step_data["actions"] = actions.reshape((1, cfg.env.num_envs, -1))
  rb.add(step_data, validate_args=cfg.buffer.validate_args)
  ```
  Note: `step_data` is added **before** stepping the env; the observation in it is the *current* obs,
  the action is the one about to be taken. After `envs.step` (`:589`), the next obs overwrites the obs
  keys in `step_data` (`:628-629`) for the next iteration.
- **Reset/done add** — `:639-650`: on any env that terminated/truncated, a separate `reset_data` dict
  (carrying the real final observation) is added via `rb.add(reset_data, dones_idxes, ...)`.
- **Sampling for training** — `:660-671`: once `iter_num >= learning_starts`, sample
  `per_rank_gradient_steps` × `(batch_size, sequence_length)` and loop `train()` over them
  (`:673-698`).

## 7. How the player loop collects and stores observations

`dreamer_v3.py:553-657` (one iteration):

1. **Get action** (`:558-584`): during prefill (`iter_num <= learning_starts`) actions are random
   (`:563`); otherwise `prepare_obs` builds a torch obs dict (`utils.py:80-91`) and
   `player.get_actions(torch_obs, mask=mask)` runs the encoder + RSSM + actor.
   - `PlayerDV3.get_actions` (`agent.py:661-691`): `embedded_obs = self.encoder(obs)` (`:678`) →
     update recurrent state → representation → actor. **This is the live inference encode** and the
     second place a `memories` key must be present (inject into `obs`/`torch_obs` here just as into
     `batch_obs` in `train`).
2. **Write `step_data`, add to buffer** (`:586-587`).
3. **Step env** (`:589-592`), handle restart-on-exception (`:595-608`), log episode info (`:610-618`).
4. **Roll obs forward** (`:621-637`): `real_next_obs` (with true final obs), then
   `step_data[k] = next_obs[k][np.newaxis]`, `obs = next_obs`, update `rewards`/`terminated`/
   `truncated`.
5. **Handle dones** (`:639-657`): add `reset_data`, zero the relevant `step_data` slots, reset player
   states for done envs (`player.init_states(dones_idxes)`).

`prepare_obs` (`utils.py:80-91`) loops over **all** keys in `obs` (`for k, v in obs.items()`), so any
extra key you put on the observation dict (e.g. `memories`) is converted to a tensor and shaped
`[1, num_envs, -1]` automatically — no special-casing needed for non-cnn keys.

---

## 8. What must change to add a `"memories"` key — checklist

| # | Change | File:line | Effort |
|---|---|---|---|
| 1 | Register `memories` in env observation space (`Box(shape=(memory_dim,))`) via wrapper | env wrapper | trivial |
| 2 | Add `"memories"` to `algo.mlp_keys.encoder`; set `algo.mlp_keys.decoder` to exclude it | config YAML | trivial |
| 3 | Seed `step_data["memories"]` in the initial block | `dreamer_v3.py:539-546` | trivial |
| 4 | Retrieve + write `step_data["memories"]` before `rb.add` | `dreamer_v3.py:586` | medium |
| 5 | Add `reset_data["memories"]` (zeros) on dones | `dreamer_v3.py:642-650` | trivial |
| 6 | Inject `memories` into `obs`/`torch_obs` before `player.get_actions` | `dreamer_v3.py:573-577` | medium |
| 7 | (No buffer code change) — flexible schema stores `memories` automatically | `buffers.py` | none |
| 8 | (No encoder code change) — `MLPEncoder` concatenates all `mlp_keys` | `agent.py:149-151` | none |

**Verdict:** confirms TODO 4.1's assessment. The buffer is schema-flexible, so storing memories is
pure plumbing in `dreamer_v3.py` + config + an env-space registration. Sequence alignment is free
because the sampler applies identical timestep indices to every key. No `buffers.py` edits needed for
the static approach.
