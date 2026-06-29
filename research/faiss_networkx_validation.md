# FAISS + NetworkX Validation (TODO 1.1a / 1.1b)

**Question:** Can FAISS (vectors) + NetworkX (graph) replace LadybugDB as the POC 1 memory backend?

**Verdict: YES.** Both work, the API is trivial, and FAISS vector search is **~12–17× faster** than LadybugDB at the scales tested. The cost is that persistence and "one system" become two systems you wire together yourself.

Environment: `faiss-cpu 1.14.3`, `networkx 3.6.1`, 768-dim CLIP-scale vectors, MacBook M5 Pro.
Script: `research/benchmark_faiss_networkx.py` → `.venv/bin/python research/benchmark_faiss_networkx.py`.

---

## 1. FAISS vector search (TODO 1.1a)

`IndexFlatIP` = exact brute-force inner product. For **unit-normalized** vectors, inner product **is** cosine similarity (higher = closer). No approximate index, no index-build step — `add()` is the whole "build".

| nodes | insert (s) | KNN k=10 mean (ms) | p50 (ms) | p95 (ms) | LadybugDB mean (ms) | speedup |
|------:|-----------:|-------------------:|---------:|---------:|--------------------:|--------:|
| 1,000 | 0.0001 | **0.064** | 0.037 | 0.041 | 1.1 | ~17× |
| 10,000 | 0.0006 | **0.368** | 0.363 | 0.401 | 4.4 | ~12× |

- Insert is effectively free (`index.add(matrix)` is a single vectorized op, not a per-row prepared statement like Ladybug's loop).
- Query scales linearly with N (it's exact brute force) but stays well under 0.5 ms at 10k. Way inside any real-time tick budget (16–33 ms).
- **No serialization friction:** FAISS consumes `np.float32` arrays directly. No Cypher, no per-row `prepare`/`execute`.

```python
import faiss, numpy as np
vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)   # cosine requires unit norm
index = faiss.IndexFlatIP(768)                         # inner product == cosine
index.add(vecs)                                        # ids are implicit row 0..n-1
D, I = index.search(query[None, :], 10)               # D=similarities desc, I=node ids
```

**Cosine vs L2:** both available. `IndexFlatIP` + normalized = cosine; `IndexFlatL2` = euclidean. We use cosine (matches Ladybug's `metric := 'cosine'`).

**ID caveat (important for integration):** `IndexFlatIP` returns *row positions* (0..n-1), not arbitrary keys. To use real node IDs or to delete vectors, wrap with `IndexIDMap` / `IndexIDMap2`. For POC 1 we map row position → graph node `s{i}` directly, which is enough. Deletion from a flat index is O(n) (rebuild); not a problem at POC scale but worth noting for pruning (TODO memory growth/pruning).

---

## 2. NetworkX graph (TODO 1.1b)

`DiGraph` with typed edges via an edge attribute `type ∈ {ACTION, RESULT, CONTEXT}`. All five required graph behaviors pass:

| test | result |
|---|---|
| Nodes carry attributes (id, prediction_error, timestamp, text_metadata) | ✅ |
| Typed edges (ACTION / RESULT / CONTEXT) | ✅ |
| Traverse ACTION edge from a node | ✅ `s10 → act10 (left)` |
| Multi-hop RESULT traversal (3 hops) | ✅ `s10 → s11 → s12 → s13` |
| Delete node → all its edges removed, no dangling refs | ✅ (`s11` had degree 16, 0 dangling after `remove_node`) |
| Pickle save → reload → graph + attrs intact | ✅ (nodes, edges, and `text_metadata` all match; 600 nodes/1189 edges = ~73 KB) |

```python
import networkx as nx
G = nx.DiGraph()
G.add_node("s10", kind="state", prediction_error=0.42,
           timestamp=1_700_000_010, text_metadata="monologue at tick 10")
G.add_node("act10", kind="action", action_type="left")
G.add_edge("s10", "act10", type="ACTION")     # state -> action
G.add_edge("s10", "s11",   type="RESULT")      # state -> next_state
G.add_edge("s10", "s3",    type="CONTEXT")     # cross-link to a similar past state

# traverse one edge type out of a node
actions = [(v, d) for _, v, d in G.out_edges("s10", data=True) if d["type"] == "ACTION"]

# delete (auto-removes incident edges)
G.remove_node("s11")

# persist
import pickle
pickle.dump(G, open("graph.pkl", "wb"))
G = pickle.load(open("graph.pkl", "rb"))
```

**Persistence is manual.** Unlike LadybugDB (columnar disk store, ACID, auto-persist), NetworkX is in-memory only — you explicitly `pickle.dump`/`load`. No transactions, no crash safety. For POC 1 (consolidate on sleep/rest) that's fine: dump the graph + FAISS index together at rest cycles. For production durability this is the main weakness vs Ladybug.

---

## 3. Integration — the core retrieval op

`query vector → FAISS nearest node IDs → NetworkX follows edges → actions/results`. Exactly the PROJECT_CONTEXT retrieval path. The two systems share a key: FAISS row position `i` ↔ graph node `s{i}`.

Query = stored vector #123 + small noise. Top hit is correctly `s123` (cosine 0.966), and the graph yields its action and result state:

```
rank 0: s123  cos=0.9658  action=right -> s124  err=0.513   <- correct nearest, edges followed
rank 1: s88   cos=0.0811  action=up    -> s89   err=0.411
rank 2: s244  cos=0.0791  action=up    -> s245  err=0.101
rank 3: s23   cos=0.0778  action=right -> s24   err=0.044
rank 4: s61   cos=0.0768  action=down  -> s62   err=0.638
```

The cosine gap (0.97 vs ~0.08 for everything else) is exactly the signal POC 1.5/1.6 need: one strong match → high confidence / low novelty; a cluster of ~0.08 noise → low confidence / high novelty falls right out of `D`.

```python
D, I = index.search(query[None, :], 10)
for vid, sim in zip(I[0], D[0]):
    node = f"s{int(vid)}"
    action = [v for _, v, d in G.out_edges(node, data=True) if d["type"] == "ACTION"]
    result = [v for _, v, d in G.out_edges(node, data=True) if d["type"] == "RESULT"]
    # -> (node, cosine=sim, action_taken, result_state, prediction_error, timestamp, text)
```

---

## FAISS+NetworkX vs LadybugDB

| dimension | FAISS + NetworkX | LadybugDB |
|---|---|---|
| Vector KNN speed (10k) | **0.37 ms** | 4.4 ms |
| Insert | trivial, vectorized | per-row prepared stmt |
| numpy in/out | direct | direct (FLOAT[768]) |
| Graph traversal | Python edge filtering | Cypher `MATCH ()-[:ACTION]->()` |
| Persistence | **manual pickle, no ACID** | auto disk, ACID, columnar |
| Systems to wire | **two (shared id key)** | one |
| Cosine support | yes (IP + normalize) | yes (`metric:='cosine'`) |
| Delete from vector index | O(n) rebuild (flat) / IDMap | native |

**What's faster:** FAISS, decisively (~12–17×), and insert is near-free.

**What's simpler:** *Per-operation*, FAISS+NetworkX — plain numpy + Python, no query language, no install of a vector extension. *System-level*, LadybugDB — one store with built-in persistence/ACID instead of two systems you keep in sync and pickle yourself.

---

## Issues encountered

- **`timeout` not on macOS** — irrelevant; the whole script runs in <1 s.
- **FAISS flat ids are row positions, not keys.** Fine when row order == node order (our case). For arbitrary IDs or vector deletion during pruning, use `IndexIDMap2`. Flat-index deletion is O(n) rebuild.
- **NetworkX has no durability story.** In-memory + manual pickle. No crash safety / ACID. Acceptable for POC (persist at rest cycles); a real risk if memory must survive hard crashes.
- **Exact search only here.** `IndexFlatIP` is brute force — perfect recall, linear scan. At 100k+ this grows; switch to `IndexHNSWFlat` or `IndexIVFFlat` (approximate, sub-ms) when scale demands. Not tested per the 10k cap.

## Recommendation

FAISS+NetworkX is a **valid, faster fallback** and satisfies every POC 1 requirement (1.1a vector search, 1.1b typed-edge traversal/CRUD/persistence). The trade is operational: you own persistence and keep two structures synced on one shared ID. For POC 1–3 (consolidate-on-sleep, no hard durability requirement) the speed and zero-friction numpy API make it attractive. LadybugDB remains the better single-system choice if ACID persistence and one-store simplicity outweigh the ~12× vector-speed gap.
