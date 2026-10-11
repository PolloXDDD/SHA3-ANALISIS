"""Verify provenance, numerical summaries and every saved attack endpoint."""
from pathlib import Path
import hashlib
import json
import sys
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from sha3lab.keccak import sha3


def load(name):
    return json.loads((ROOT/"results"/(name+".json")).read_text())


def independent_cycles(mapping):
    """Functional-graph pruning, independent of Tarjan."""
    n = len(mapping)
    indegree = np.bincount(mapping,minlength=n)
    queue = list(np.flatnonzero(indegree==0))
    for v in queue:
        u = int(mapping[v])
        indegree[u] -= 1
        if indegree[u]==0:
            queue.append(u)
    alive = set(np.flatnonzero(indegree>0).tolist())
    lengths = []
    while alive:
        v = next(iter(alive))
        count = 0
        while v in alive:
            alive.remove(v)
            count += 1
            v = int(mapping[v])
        lengths.append(count)
    return sorted(lengths)


def main():
    manifest = load("manifest")
    assert not manifest["quick"]
    for name,expected in manifest["source_sha256"].items():
        assert hashlib.sha256((ROOT/"sha3lab"/name).read_bytes()).hexdigest()==expected,name
    cert = load("collision_certificate")
    a,b = [bytes.fromhex(cert[f"message_{v}_hex"]) for v in ("a","b")]
    assert a != b
    assert sha3(a,rounds=1).hex()==sha3(b,rounds=1).hex()==cert["reduced_digest_a"]==cert["reduced_digest_b"]
    for name,msg in (("a",a),("b",b)):
        assert hashlib.sha3_256(msg).hexdigest()==cert[f"full_sha3_256_{name}"]
    diffusion = load("diffusion")
    arrays = np.load(ROOT/"results/diffusion.npz")
    for name in ("kernel","control"):
        for part in ("state","digest"):
            raw = arrays[f"{part}_weights_{name}"]
            assert raw.shape == (8192,25)
            for r in (1,2,3,4,6,12,24):
                assert abs(raw[:,r].mean()-diffusion[f"{name}_{r}"][part]["mean"])<1e-12
                if part=="digest":
                    assert int(np.sum(raw[:,r]==0))==diffusion[f"{name}_{r}"]["digest_collisions"]
    markov = load("markov")
    losses = np.load(ROOT/"results/markov_losses.npz")
    for key in losses.files:
        assert abs(losses[key].mean()-markov[key]["mean"])<1e-12
    maps = np.load(ROOT/"results/projected_maps.npz")
    cycles = load("projected_cycles")
    for r in (1,2,12):
        assert independent_cycles(maps[f"mapping_{r}"])==cycles[str(r)]["cycle_lengths"]
    total_endpoints = 0
    for dataset,summary_name in (("search_trials","search_summary"),("ablation_trials","ablation_summary")):
        rows,summary = load(dataset),load(summary_name)
        for row in rows:
            message = bytes.fromhex(row["message_hex"])
            assert len(message)==128
            digest = hashlib.sha3_256(message).digest() if row["rounds"]==24 else sha3(message,rounds=row["rounds"])
            value = int.from_bytes(digest[:2],"little") & 4095
            assert row["value"]==value
            assert row["best_hamming"]==(value ^ row["target"]).bit_count()
            assert row["success"]==(value==row["target"])
            assert 1 <= row["queries"] <= summary["settings"]["budget"]
            total_endpoints += 1
        for r in {row["rounds"] for row in rows}:
            for method in {row["method"] for row in rows}:
                selected = [row for row in rows if row["rounds"]==r and row["method"]==method]
                assert len(selected)==summary[str(r)][method]["trials"]==48
                assert sum(row["success"] for row in selected)==summary[str(r)][method]["successes"]
    suite = unittest.defaultTestLoader.discover(str(ROOT/"tests"))
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    assert result.wasSuccessful()
    report = {"status":"PASS","test_methods":result.testsRun,
        "saved_search_endpoints_checked":total_endpoints,"source_fingerprints_checked":len(manifest["source_sha256"]),
        "diffusion_summaries":"recomputed from raw arrays","projected_cycles":"independently checked by indegree pruning",
        "collision_certificate":"verified reduced digests and hashlib full-round digests"}
    (ROOT/"results/verification.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))


if __name__ == "__main__":
    main()
