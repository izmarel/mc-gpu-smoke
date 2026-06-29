# LadybugDB API Verification (TODO 1.1a + 1.1b)

**Status: VERIFIED by running real code. Verdict: USE IT. No FAISS fallback needed for our scale.**

- Package: `ladybug` **0.17.1** (`pip install ladybug`), self-described as *"Highly scalable, extremely fast, easy-to-use embeddable graph database."*
- Python: **3.12.13** in `.venv` (note: not 3.9 — venv is 3.12).
- Platform tested: MacBook (Apple Silicon), darwin.
- This is the Kuzu engine, rebranded. The Python API (`Database`, `Connection`, `QueryResult`, `prepare`/`execute`, Cypher dialect, `CALL CREATE_VECTOR_INDEX`) is byte-for-byte Kuzu. Anything written for Kuzu applies.

Every code snippet below was executed. Benchmark numbers are from `research/benchmark_ladybug.py` (committed alongside this file), run on this machine at 768-dim (CLIP ViT-B/32 scale).

---

## TL;DR for the project

| Question | Answer |
|---|---|
| Vector nodes in Cypher? | Yes — `FLOAT[768]` fixed-size array property. |
| KNN search syntax? | `CALL QUERY_VECTOR_INDEX('Table','idx', $q, k) RETURN node.*, distance` |
| Cosine? | Yes. Also `l2`, `l2sq`, `dotproduct`. Pick `cosine`. |
| Python API? | `conn.execute(cypher, {params}).get_all()` → list of lists. |
| Speed at 1k/10k/100k? | KNN query **1.1 / 4.7 / 10.6 ms** mean (k=10, 768-dim). Fits a 16–33 ms tick. |
| numpy direct? | Yes — pass `np.ndarray` (float32 or float64) straight as a param. Returns as Python `list`. |
| Typed edges (ACTION/RESULT/CONTEXT)? | Yes — `CREATE REL TABLE ACTION(FROM ... TO ...)`. |
| Multi-hop traversal? | Yes — including variable-length `-[:RESULT*1..5]->`. |
| Persistence? | Automatic. Single-file DB, auto-commit. No explicit flush. |
| Index after restart? | Survives. Must re-run `LOAD vector;` per connection. |
| Index sees new inserts? | **Yes, incrementally — no rebuild needed.** Deletes also reflected. |
| CRUD? | Full — `SET`, `DELETE`, `DETACH DELETE`. |

**One real gotcha:** table names are **case-insensitive and share one namespace**. A REL table named `ACTION` collides with a NODE table named `Action` ("already exists in catalog"). Name node tables and rel tables distinctly (e.g. node `ActionNode` + rel `ACTION`).

---

## VECTOR SEARCH

### Creating nodes with vector properties

Vectors are a fixed-size float array column on a node table: `FLOAT[D]`.

```python
import ladybug, numpy as np
db = ladybug.Database("memory.db")          # single file, created if absent
conn = ladybug.Connection(db)
conn.execute("INSTALL vector; LOAD vector;") # vector funcs live in an extension

conn.execute("CREATE NODE TABLE Mem(id INT64, emb FLOAT[768], err DOUBLE, ts INT64, txt STRING, PRIMARY KEY(id))")

# Insert. numpy array passes straight through as a parameter — no serialization.
v = np.random.randn(768).astype(np.float32)
v /= np.linalg.norm(v)
conn.execute("CREATE (:Mem {id: $id, emb: $emb, err: $e, ts: $t, txt: $x})",
             {"id": 1, "emb": v, "e": 0.42, "t": 1719500000, "x": "saw a door"})
```

Verified: `np.float32` **and** `np.float64` ndarrays both accepted directly; so is a plain Python list. On read, the vector comes back as a Python `list[float]` (wrap in `np.asarray(...)` if you need an array).

### KNN similarity search

Build an index once, then call `QUERY_VECTOR_INDEX`:

```python
conn.execute("CALL CREATE_VECTOR_INDEX('Mem', 'memidx', 'emb', metric := 'cosine')")

q = np.random.randn(768).astype(np.float32); q /= np.linalg.norm(q)
res = conn.execute(
    "CALL QUERY_VECTOR_INDEX('Mem', 'memidx', $q, 10) "
    "RETURN node.id, node.err, node.txt, distance ORDER BY distance",
    {"q": q})
for row in res.get_all():
    print(row)   # [id, err, txt, distance]
```

`node` is the bound matched node — you can return any of its properties **and** the `distance` in the same call, so retrieval + metadata fetch is one query. Lower `distance` = closer (for cosine, `distance ≈ 1 - cosine_similarity`; identical unit vectors → ~0).

### Cosine vs L2

All four metrics create successfully (verified by building an index with each):

```
cosine     OK
l2         OK
l2sq       OK
dotproduct OK
```

Use `metric := 'cosine'` — matches CLIP/embedding convention and our unit-normalized state vectors.

### Python API for executing + reading results

`conn.execute(cypher, params_dict)` → `QueryResult`. Result readers, all verified:

```python
r = conn.execute("MATCH (m:Mem) RETURN m.id AS id, m.txt AS txt ORDER BY m.id")
r.get_all()           # -> [[0,'node0'], [2,'node2']]      (list of lists)
r.get_column_names()  # -> ['id','txt']
# rows_as_dict() -> iterator of dicts:
list(conn.execute("MATCH (m:Mem) RETURN m.id AS id, m.txt AS txt").rows_as_dict())
#   -> [{'id':0,'txt':'node0'}, {'id':2,'txt':'node2'}]
# streaming:
r = conn.execute("MATCH (m:Mem) RETURN m.id")
while r.has_next(): print(r.get_next())   # one row (list) at a time
```

`get_as_df()` / `get_as_pl()` exist but need `pandas` / `polars` installed (not in venv — `ModuleNotFoundError: No module named 'pandas'`). Not needed; `get_all()` / `rows_as_dict()` cover us.

### Benchmark — speed at 1k / 10k / 100k (768-dim, cosine, k=10)

Real numbers from `research/benchmark_ladybug.py` on this machine. Query latency averaged over 100 random queries per size.

```
   nodes  insert(s)   index(s)  q_mean(ms)  q_p50(ms)  q_p95(ms)
    1000       4.29       1.78       1.115      1.093      1.269
   10000      45.27       8.12       4.683      4.645      5.047
  100000     441.46      81.04      10.577     10.457     11.914
```

Reading the numbers:
- **Query latency is the thing that matters for the tick loop, and it's fine.** 10.6 ms mean at 100k nodes, p95 ~12 ms. A 30–60 fps tick has a 16–33 ms budget; KNN retrieval fits with room to spare even at 100k memories. At POC-1 scale (≤10k) it's ~5 ms.
- **Insert is ~4.4 ms per node, flat across scale** (4.29s/1k, 45s/10k, 441s/100k). That's row-by-row single inserts — which is exactly our access pattern (one experience stored per tick), so per-insert cost is what counts, and ~4 ms/insert is acceptable. If we ever need to bulk-load, `COPY ... FROM` a file/arrow is the fast path (not benchmarked here).
- **Index build is one-time and cheap** relative to insert (81s to index 100k). And per the incremental finding below, we rarely rebuild.

Honest caveat: insert throughput via single `execute()` calls is modest (~225 rows/s). Not a problem for real-time single-experience storage; would be a problem for loading a 1M-row dataset row-by-row (use COPY there).

### numpy arrays — direct or serialized?

**Direct.** No serialization layer needed.

```python
arr32 = np.random.randn(768).astype(np.float32)
conn.execute("CREATE (:Mem {id:$i, emb:$e})", {"i":1, "e":arr32})        # OK
conn.execute("CREATE (:Mem {id:$i, emb:$e})", {"i":2, "e":arr32.tolist()})# OK
conn.execute("CREATE (:Mem {id:$i, emb:$e})", {"i":3, "e":arr32.astype(np.float64)}) # OK
```

All three forms verified. Storage column is `FLOAT[768]` (single precision), so float64 input is narrowed to float32 — fine for embeddings.

---

## GRAPH TRAVERSAL (TODO 1.1b)

### Typed / labelled edges (ACTION, RESULT, CONTEXT)

Yes. Each edge type is a REL TABLE with declared endpoints.

```python
conn.execute("CREATE NODE TABLE State(id INT64, emb FLOAT[768], err DOUBLE, PRIMARY KEY(id))")
conn.execute("CREATE NODE TABLE ActionNode(id INT64, kind STRING, PRIMARY KEY(id))")

conn.execute("CREATE REL TABLE ACTION(FROM State TO ActionNode)")  # state -> action taken
conn.execute("CREATE REL TABLE RESULT(FROM State TO State)")       # state -> next state
conn.execute("CREATE REL TABLE CONTEXT(FROM State TO State)")      # state -> co-active memory
```

> ⚠️ **Namespace collision:** `CREATE REL TABLE ACTION(...)` fails with *"ACTION already exists in catalog"* if a node table named `Action` exists — names are case-insensitive and node+rel share one catalog. Hence `ActionNode` for the node, `ACTION` for the edge. Hit this for real during testing.

Create edges by matching the endpoints:

```python
conn.execute("MATCH (s:State),(a:ActionNode) WHERE s.id=1 AND a.id=10 CREATE (s)-[:ACTION]->(a)")
conn.execute("MATCH (a:State),(b:State) WHERE a.id=1 AND b.id=2 CREATE (a)-[:RESULT]->(b)")
```

Edges can carry properties too (`CREATE REL TABLE ACTION(FROM State TO ActionNode, weight DOUBLE)`), useful later for edge-strengthening during skill compression.

### Traversing edges with Cypher

```python
# one hop along a specific edge type
conn.execute("MATCH (s:State)-[:ACTION]->(a:ActionNode) WHERE s.id=1 RETURN a.kind").get_all()
# -> [['move_left']]
```

### Multi-hop (state → action → result → next_state)

Both fixed multi-hop and variable-length verified:

```python
# state -> result(next state) -> action taken in that next state
conn.execute(
  "MATCH (s:State)-[:RESULT]->(n:State)-[:ACTION]->(a:ActionNode) "
  "WHERE s.id=1 RETURN s.id, n.id, a.kind").get_all()
# -> [[1, 2, 'move_right']]

# variable-length: follow the RESULT chain 1..5 hops out (whole episode)
conn.execute(
  "MATCH (s:State)-[:RESULT*1..5]->(n:State) WHERE s.id=1 RETURN n.id ORDER BY n.id").get_all()
# -> [[2], [3]]
```

`-[:RESULT*1..5]->` is exactly the episode-walk the architecture needs (state → ... → terminal). You can also bind the whole path with `MATCH p = (...)` and inspect it.

### Persistence — auto-save or explicit flush?

**Automatic.** Wrote 5 nodes + a vector index, called **no** flush/commit, closed, reopened in a **fresh process** — all 5 nodes and the index were intact.

```python
# process A
db = ladybug.Database("memory.db"); conn = ladybug.Connection(db)
... # CREATE nodes, CREATE_VECTOR_INDEX
conn.close(); db.close()         # no explicit flush anywhere

# process B (separate run)
db = ladybug.Database("memory.db"); conn = ladybug.Connection(db)
conn.execute("LOAD vector;")     # <-- must reload the extension each connection
conn.execute("MATCH (m:Mem) RETURN count(m)").get_all()   # -> [[5]]
conn.execute("CALL QUERY_VECTOR_INDEX('Mem','idx',$q,3) RETURN node.id, distance", {"q":[...]})
# -> index still works, returns neighbors
```

Notes:
- The DB is a **single file** on disk (102 KB for the tiny test). Not a directory.
- Writes auto-commit (each `execute` is its own transaction unless you `BEGIN TRANSACTION`). ACID per the engine.
- The **only** thing you must repeat after reopening is `LOAD vector;` — the extension *install* persists machine-wide, but it must be *loaded* into each new connection before you call vector functions.

### Update / delete nodes and edges

Full CRUD, all verified:

```python
conn.execute("MATCH (s:State) WHERE s.id=1 SET s.err = 0.99")              # update property
conn.execute("MATCH (s:State)-[r:RESULT]->(n) WHERE s.id=2 DELETE r")       # delete edge
conn.execute("MATCH (a:ActionNode) WHERE a.id=11 DETACH DELETE a")          # delete node + its edges
```

`DELETE` on a node with edges errors; use `DETACH DELETE` to drop the node and its relationships together.

**Index reflects mutations live** — after `DETACH DELETE` of an indexed node, KNN stops returning it (verified: deleted id no longer in results). This matters for the prune/compress step.

### Incremental index growth (critical for an always-growing memory graph)

The big risk was "do I have to rebuild the HNSW index every time I add a memory?" **No.**

```python
# index built over ids {0,1,2}, THEN insert id=99 close to the query, NO rebuild:
conn.execute("CREATE (:Mem {id:99, emb:[5.0,0.0,0.0]})")
conn.execute("CALL QUERY_VECTOR_INDEX('Mem','idx',$q,5) RETURN node.id ORDER BY distance",
             {"q":[5.0,0.0,0.0]})
# -> [[99, ...], [1, ...], [2, ...], [0, ...]]   id=99 is returned, top-ranked
```

New nodes are searchable immediately without `DROP_VECTOR_INDEX` + rebuild. (`DROP_VECTOR_INDEX` + re-`CREATE_VECTOR_INDEX` exists if you ever want a clean compaction, e.g. during sleep.) This kills the main scaling worry for POC 1.

---

## Decision: LadybugDB stands. FAISS fallback not needed.

The TODO's fallback condition was "if LadybugDB vector search is too slow at scale." It isn't — 10.6 ms mean KNN at 100k nodes, 768-dim, single embedded process, fits the tick budget. We get vectors + typed graph + multi-hop traversal + auto-persistence + incremental index from one `pip install`, no FAISS / NetworkX / SQLite glue. Recommend locking 1.1 from *tentative* to *decided*.

### Things to remember when building POC 1 storage layer
1. `LOAD vector;` on every fresh connection before any vector call.
2. Distinct catalog names: node `ActionNode` vs rel `ACTION` (case-insensitive namespace).
3. Pass numpy float32 arrays directly; expect Python `list` back.
4. Unit-normalize state vectors + `metric := 'cosine'`.
5. One experience = one `execute` insert (~4 ms); fine for real-time. Use `COPY FROM` only for bulk dataset loads.
6. Index is incremental for inserts/deletes — rebuild only as a deliberate sleep-time compaction.
7. DB is a single file; back it up by copying that file.

### Versions / repro
- `ladybug==0.17.1`, Python 3.12.13, darwin/Apple Silicon.
- Benchmark: `.venv/bin/python research/benchmark_ladybug.py` (DIM=768, k=10, 100 queries/size).
- A `DeprecationWarning` notes `prepare()` + `execute()` is deprecated in favor of a single `execute(query, params)` call — use the one-call form in project code.
