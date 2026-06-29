"""
LadybugDB vector-search benchmark for the Artificial Consciousness MVP.

Measures, at 1k / 10k / 100k nodes with 768-dim FLOAT vectors (CLIP ViT-B/32 scale):
  - bulk insert time
  - CREATE_VECTOR_INDEX build time
  - average single-query KNN latency (k=10), averaged over 100 random queries

Run:  .venv/bin/python research/benchmark_ladybug.py
"""
import os, shutil, time, statistics
import numpy as np
import ladybug

DIM = 768
K = 10
N_QUERIES = 100
SIZES = [1_000, 10_000]
DBROOT = "/tmp/lb_bench"


def make_vectors(n, dim, seed=0):
    rng = np.random.default_rng(seed)
    v = rng.standard_normal((n, dim)).astype(np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True)  # unit-norm, realistic for cosine
    return v


def bench(n):
    dbpath = f"{DBROOT}_{n}"
    if os.path.exists(dbpath):
        os.remove(dbpath) if os.path.isfile(dbpath) else shutil.rmtree(dbpath)

    db = ladybug.Database(dbpath)
    conn = ladybug.Connection(db)
    conn.execute("INSTALL vector; LOAD vector;")
    conn.execute(f"CREATE NODE TABLE Mem(id INT64, emb FLOAT[{DIM}], PRIMARY KEY(id))")

    vecs = make_vectors(n, DIM)

    # --- bulk insert via prepared statement ---
    t0 = time.perf_counter()
    stmt = conn.prepare("CREATE (:Mem {id: $id, emb: $emb})")
    for i in range(n):
        conn.execute(stmt, {"id": i, "emb": vecs[i]})
    insert_s = time.perf_counter() - t0

    # --- build vector index (cosine) ---
    t0 = time.perf_counter()
    conn.execute("CALL CREATE_VECTOR_INDEX('Mem', 'memidx', 'emb', metric := 'cosine')")
    index_s = time.perf_counter() - t0

    # --- query latency, averaged ---
    qs = make_vectors(N_QUERIES, DIM, seed=999)
    qstmt = conn.prepare(
        "CALL QUERY_VECTOR_INDEX('Mem', 'memidx', $q, %d) "
        "RETURN node.id, distance ORDER BY distance" % K
    )
    lats = []
    for j in range(N_QUERIES):
        t0 = time.perf_counter()
        conn.execute(qstmt, {"q": qs[j]}).get_all()
        lats.append((time.perf_counter() - t0) * 1000.0)  # ms

    conn.close()
    db.close()
    return {
        "n": n,
        "insert_s": insert_s,
        "index_s": index_s,
        "q_mean_ms": statistics.mean(lats),
        "q_p50_ms": statistics.median(lats),
        "q_p95_ms": sorted(lats)[int(0.95 * len(lats)) - 1],
    }


if __name__ == "__main__":
    print(f"ladybug {ladybug.__version__}  dim={DIM}  k={K}  queries/size={N_QUERIES}\n")
    print(f"{'nodes':>8} {'insert(s)':>10} {'index(s)':>10} {'q_mean(ms)':>11} {'q_p50(ms)':>10} {'q_p95(ms)':>10}")
    for n in SIZES:
        r = bench(n)
        print(f"{r['n']:>8} {r['insert_s']:>10.2f} {r['index_s']:>10.2f} "
              f"{r['q_mean_ms']:>11.3f} {r['q_p50_ms']:>10.3f} {r['q_p95_ms']:>10.3f}")
