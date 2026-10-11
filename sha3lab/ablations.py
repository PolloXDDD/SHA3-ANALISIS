"""Independent-seed replication and attribution controls for search.

The main experiment suggested a two-round improvement over random search.
This follow-up fixes the algorithm and uses new seeds/targets. Hill climbing
uses uniform actions; no_feedback fixes the self-feedback bit; no_scc replaces
the graph test with the same fixed stagnation restart trigger. Local methods
use common seeds within each target. No settings are tuned on these outcomes.
"""
import argparse
import json
import time
from pathlib import Path
import numpy as np
from scipy.stats import binomtest
from .attacks import truncated_preimage
from .experiments import dump,wilson,SEED


def run(out,trials=48,budget=2048):
    start = time.perf_counter()
    rng = np.random.default_rng(SEED+9000000)
    methods = ("random","hillclimb","meta_no_feedback","meta_no_scc","meta")
    all_rows,summary = [],{}
    for rounds in (2,24):
        groups = {m:[] for m in methods}
        for trial in range(trials):
            target = int(rng.integers(4096))
            seed = SEED+10000000+rounds*10000+trial
            for method in methods:
                ans = truncated_preimage(target,rounds,seed,budget,12,method)
                ans.update({"rounds":rounds,"trial":trial,"seed":seed,"target":target,"method":method})
                groups[method].append(ans)
                all_rows.append(ans)
            if (trial+1) % 8 == 0:
                print(f"replication r={rounds}: {trial+1}/{trials}",flush=True)
        result = {}
        for method,rows in groups.items():
            hits = sum(row["success"] for row in rows)
            result[method] = {"successes":hits,"trials":trials,"wilson_ci95":wilson(hits,trials),
                "mean_queries_censored":float(np.mean([row["queries"] for row in rows]))}
        comparisons = {}
        for method in methods[:-1]:
            diffs = [int(a["success"])-int(b["success"]) for a,b in zip(groups["meta"],groups[method])]
            plus,minus = diffs.count(1),diffs.count(-1)
            comparisons[method] = {"meta_wins":plus,"other_wins":minus,
                "exact_p":float(binomtest(plus,plus+minus,.5).pvalue) if plus+minus else 1.0,
                "identical_endpoints":sum(a["message_hex"]==b["message_hex"] and a["queries"]==b["queries"]
                                         for a,b in zip(groups["meta"],groups[method]))}
        result["paired_comparisons"] = comparisons
        summary[str(rounds)] = result
    summary["settings"] = {"trials":trials,"budget":budget,"target_bits":12,
        "target_seed":SEED+9000000,"selection":"follow-up after initial experiment; algorithm unchanged",
        "elapsed_seconds":time.perf_counter()-start}
    out.mkdir(parents=True,exist_ok=True)
    dump(out/"ablation_trials.json",all_rows)
    dump(out/"ablation_summary.json",summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out",type=Path,default=Path("results"))
    args = parser.parse_args()
    run(args.out)
