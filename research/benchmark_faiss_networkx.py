"""
FAISS + NetworkX validation for the Artificial Consciousness MVP memory backend.

Tests whether FAISS (vectors) + NetworkX (graph) can replace LadybugDB.

  1. FAISS IndexFlatIP (cosine via normalized vectors), 768-dim, 1k + 10k nodes:
     insert time, KNN(k=10) query latency averaged over 100 queries.
  2. NetworkX DiGraph: typed edges (ACTION/RESULT/CONTEXT), traversal,
     multi-hop, node deletion, pickle persistence.
  3. Integration: query vector -> FAISS nearest node IDs -> NetworkX edges
     -> actions/results. The core retrieval op.

Run:  .venv/bin/python research/benchmark_faiss_networkx.py
"""
import os, time, pickle, statistics
import numpy as np
import faiss
import networkx as nx

DIM = 768
K = 10
N_QUERIES = 100
SIZES = [1_000, 10_000]
PKL = "/tmp/fn_graph.pkl"


def make_vectors(n, dim, seed=0):
    rng = np.random.default_rng(seed)
    v = rng.standard_normal((n, dim)).astype(np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True)  # unit-norm => inner product == cosine
    return v


# ---------------------------------------------------------------- 1. FAISS
def bench_faiss(n):
    vecs = make_vectors(n, DIM)

    t0 = time.perf_counter()
    index = faiss.IndexFlatIP(DIM)   # inner product = cosine for normalized vectors
    index.add(vecs)                  # ids are implicit 0..n-1 (row order)
    insert_s = time.perf_counter() - t0

    qs = make_vectors(N_QUERIES, DIM, seed=999)
    lats = []
    for j in range(N_QUERIES):
        q = qs[j:j+1]                # (1, DIM)
        t0 = time.perf_counter()
        D, I = index.search(q, K)    # D=similarities (desc), I=node ids
        lats.append((time.perf_counter() - t0) * 1000.0)
    return {
        "n": n,
        "insert_s": insert_s,
        "q_mean_ms": statistics.mean(lats),
        "q_p50_ms": statistics.median(lats),
        "q_p95_ms": sorted(lats)[int(0.95 * len(lats)) - 1],
        "index": index,
    }


# ---------------------------------------------------------------- 2. NetworkX
def build_graph(n_states=300):
    """Build state->action->result->next_state chains with typed edges + CONTEXT cross-links."""
    G = nx.DiGraph()
    rng = np.random.default_rng(7)
    # node ids: state nodes s0..; action nodes a0..; we interleave a linear episode
    # chain: s0 -ACTION-> act0 ; s0 -RESULT-> s1 ; s1 -ACTION-> act1 ; s1 -RESULT-> s2 ...
    for i in range(n_states):
        G.add_node(f"s{i}", kind="state", vec_id=i,
                   prediction_error=float(rng.random()),
                   timestamp=1_700_000_000 + i,
                   text_metadata=f"monologue at tick {i}")
    for i in range(n_states):
        G.add_node(f"act{i}", kind="action", action_type=["up", "down", "left", "right"][i % 4])
        G.add_edge(f"s{i}", f"act{i}", type="ACTION")
        if i + 1 < n_states:
            G.add_edge(f"s{i}", f"s{i+1}", type="RESULT")   # state -> next_state
    # CONTEXT cross-links: each state linked to a few earlier similar states
    for i in range(5, n_states):
        for j in rng.choice(i, size=2, replace=False):
            G.add_edge(f"s{i}", f"s{int(j)}", type="CONTEXT")
    return G


def out_edges_of_type(G, node, etype):
    return [(v, d) for _, v, d in G.out_edges(node, data=True) if d.get("type") == etype]


def test_networkx():
    res = {}
    G = build_graph()
    res["n_nodes"] = G.number_of_nodes()
    res["n_edges"] = G.number_of_edges()

    # traverse ACTION edges from s10
    actions = out_edges_of_type(G, "s10", "ACTION")
    res["action_from_s10"] = [(v, G.nodes[v]["action_type"]) for v, _ in actions]

    # multi-hop: state -> action (ACTION) + state -> next_state (RESULT), 3 RESULT hops
    path = ["s10"]
    cur = "s10"
    for _ in range(3):
        nxt = out_edges_of_type(G, cur, "RESULT")
        if not nxt:
            break
        cur = nxt[0][0]
        path.append(cur)
    res["result_3hop_path"] = path

    # delete a node, verify edges gone
    s11_edges_before = G.degree("s11")
    G.remove_node("s11")
    res["s11_in_graph_after_delete"] = G.has_node("s11")
    res["s11_edges_before_delete"] = s11_edges_before
    res["dangling_edges_to_s11"] = sum(1 for u, v in G.edges() if v == "s11" or u == "s11")

    # rebuild fresh graph for persistence test (so deletion doesn't affect it)
    G2 = build_graph()
    with open(PKL, "wb") as f:
        pickle.dump(G2, f)
    with open(PKL, "rb") as f:
        G3 = pickle.load(f)
    res["pickle_nodes_match"] = G2.number_of_nodes() == G3.number_of_nodes()
    res["pickle_edges_match"] = G2.number_of_edges() == G3.number_of_edges()
    res["pickle_attrs_match"] = G3.nodes["s10"]["text_metadata"] == G2.nodes["s10"]["text_metadata"]
    res["pickle_size_kb"] = os.path.getsize(PKL) / 1024.0
    return res


# ---------------------------------------------------------------- 3. Integration
def test_integration():
    """FAISS finds nearest state node ids; NetworkX follows their edges to actions/results."""
    n = 300
    vecs = make_vectors(n, DIM, seed=42)
    index = faiss.IndexFlatIP(DIM)
    index.add(vecs)

    G = build_graph(n_states=n)   # vec_id i corresponds to node s{i}

    # query = a perturbed copy of stored vector 123 -> should retrieve s123 as top hit
    q = vecs[123:124].copy()
    q += 0.01 * np.random.default_rng(1).standard_normal((1, DIM)).astype(np.float32)
    q /= np.linalg.norm(q)
    D, I = index.search(q, K)

    hits = []
    for rank, (vid, sim) in enumerate(zip(I[0], D[0])):
        node = f"s{int(vid)}"
        action = out_edges_of_type(G, node, "ACTION")
        result = out_edges_of_type(G, node, "RESULT")
        hits.append({
            "rank": rank,
            "node": node,
            "cosine": round(float(sim), 4),
            "action": G.nodes[action[0][0]]["action_type"] if action else None,
            "result_state": result[0][0] if result else None,
            "pred_error": round(G.nodes[node]["prediction_error"], 3),
        })
    return {"top_hit": hits[0], "all_hits": hits}


if __name__ == "__main__":
    print(f"faiss {faiss.__version__}  networkx {nx.__version__}  dim={DIM}  k={K}  queries/size={N_QUERIES}\n")

    print("=== 1. FAISS ===")
    print(f"{'nodes':>8} {'insert(s)':>10} {'q_mean(ms)':>11} {'q_p50(ms)':>10} {'q_p95(ms)':>10}")
    for n in SIZES:
        r = bench_faiss(n)
        print(f"{r['n']:>8} {r['insert_s']:>10.4f} {r['q_mean_ms']:>11.4f} "
              f"{r['q_p50_ms']:>10.4f} {r['q_p95_ms']:>10.4f}")

    print("\n=== 2. NetworkX ===")
    g = test_networkx()
    for k, v in g.items():
        print(f"  {k}: {v}")

    print("\n=== 3. Integration (query vec -> FAISS ids -> NetworkX edges) ===")
    it = test_integration()
    print(f"  top hit: {it['top_hit']}")
    for h in it["all_hits"][:5]:
        print(f"    rank {h['rank']}: {h['node']} cos={h['cosine']} "
              f"action={h['action']} -> {h['result_state']} err={h['pred_error']}")

    print("\nDONE")
