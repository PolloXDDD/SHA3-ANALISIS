"""Run: python -m sha3lab.experiments --out results [--quick]."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import platform
import sys
import time
from pathlib import Path
import numpy as np
from scipy import stats
from .keccak import (messages128_states, round_batch, permute_batch, bits, weight)
from .hypercomplex import grade_features
from .graphs import tarjan, dependency_graph, chi_relation_analysis
from .metamarkov import DyadicMetaMarkov, quantize, control_losses
from .attacks import collision_certificate, truncated_preimage

SEED = 20261011
CHECKPOINTS = (1,2,3,4,6,12,24)


def dump(path, value):
    Path(path).write_text(json.dumps(value,indent=2,sort_keys=True)+"\n")


def mean_ci(values):
    x = np.asarray(values,dtype=float)
    mean = float(x.mean())
    error = float(stats.t.ppf(.975,len(x)-1)*x.std(ddof=1)/math.sqrt(len(x))) if len(x)>1 else 0
    return {"mean":mean,"ci95":[mean-error,mean+error],"n":len(x)}


def auc(labels,scores):
    labels = np.asarray(labels)
    n1 = int(labels.sum())
    n0 = len(labels)-n1
    return float((stats.rankdata(scores)[labels == 1].sum()-n1*(n1+1)/2)/(n1*n0))


def auc_ci(labels,scores,rng,repeats=300):
    a,b = scores[labels==1],scores[labels==0]
    ys = np.r_[np.ones(len(a)),np.zeros(len(b))]
    values = [auc(ys,np.r_[rng.choice(a,len(a)),rng.choice(b,len(b))]) for _ in range(repeats)]
    return {"auc":auc(labels,scores),"ci95":np.quantile(values,[.025,.975]).tolist()}


def ridge_scores(a,b,cols):
    """Train on the first half of each class; test on the second half."""
    n = min(len(a),len(b))//2
    x = np.r_[a[:n,cols],b[:n,cols]]
    test = np.r_[a[n:2*n,cols],b[n:2*n,cols]]
    y = np.r_[np.ones(n),-np.ones(n)]
    mu,sd = x.mean(axis=0),np.maximum(x.std(axis=0),1e-8)
    x = np.c_[np.ones(2*n),(x-mu)/sd]
    test = np.c_[np.ones(2*n),(test-mu)/sd]
    penalty = np.eye(x.shape[1])*1.0
    penalty[0,0] = 0
    beta = np.linalg.solve(x.T@x+penalty,x.T@y)
    return (y+1)/2,test@beta


def avalanche(out,n,features_n):
    rng = np.random.default_rng(SEED+1)
    messages = rng.integers(0,256,(n,128),dtype=np.uint8)
    original = messages128_states(messages)
    kernel = original.copy()
    kernel[:,5] ^= np.uint64(1)
    kernel[:,10] ^= np.uint64(1)
    control = original.copy()
    control[:,0] ^= np.uint64(1)
    control[:,1] ^= np.uint64(1)
    arrays,summary,features = {},{},{}
    states = {"kernel":kernel,"control":control}
    state_weights = {k:np.zeros((n,25),dtype=np.uint16) for k in states}
    digest_weights = {k:np.zeros((n,25),dtype=np.uint16) for k in states}
    for name,a in states.items():
        state_weights[name][:,0] = weight(a ^ original)
        digest_weights[name][:,0] = weight((a ^ original)[:,:4])
    for r in range(1,25):
        original = round_batch(original,r-1)
        for name in states:
            states[name] = round_batch(states[name],r-1)
            d = states[name] ^ original
            state_weights[name][:,r] = weight(d)
            digest_weights[name][:,r] = weight(d[:,:4])
            if r in CHECKPOINTS:
                summary[f"{name}_{r}"] = {"state":mean_ci(state_weights[name][:,r]),
                    "digest":mean_ci(digest_weights[name][:,r]),
                    "digest_collisions":int(np.sum(digest_weights[name][:,r] == 0))}
            if name == "kernel" and r in CHECKPOINTS:
                dd = d[:features_n]
                null = rng.bit_generator.random_raw((features_n,25))
                features[f"keccak_{r}"] = np.c_[weight(dd),grade_features(dd)]
                features[f"null_{r}"] = np.c_[weight(null),grade_features(null)]
            if r == 24:
                db = bits(d[:,:4]).reshape(n,256)
                counts = db.sum(axis=0)
                p = np.array([stats.binomtest(int(c),n,.5).pvalue for c in counts])
                arrays[f"bit_counts_{name}"] = counts
                arrays[f"bit_pvalues_{name}"] = p
        if r in CHECKPOINTS:
            print(f"diffusion round {r}/24",flush=True)
    pp = np.r_[arrays["bit_pvalues_kernel"],arrays["bit_pvalues_control"]]
    ordered = np.sort(pp)
    rejected = 0
    for i,p in enumerate(ordered):
        if p > .05/(len(pp)-i):
            break
        rejected += 1
    summary["full_round_bit_tests"] = {"tests":len(pp),"holm_rejections_0.05":rejected,
        "minimum_raw_p":float(pp.min()),"family":"2 differences x 256 output bits"}
    for name in states:
        arrays[f"state_weights_{name}"] = state_weights[name]
        arrays[f"digest_weights_{name}"] = digest_weights[name]
    np.savez_compressed(out/"diffusion.npz",**arrays)
    np.savez_compressed(out/"hypercomplex_features.npz",**features)
    dump(out/"diffusion.json",summary)
    classifier = {}
    scores_archive = {}
    for r in CHECKPOINTS:
        classifier[str(r)] = {}
        for name,cols in (("hamming",[0]),("clifford_grades",list(range(1,8))),
                          ("combined",list(range(8)))):
            labels,scores = ridge_scores(features[f"keccak_{r}"],features[f"null_{r}"],cols)
            classifier[str(r)][name] = auc_ci(labels,scores,rng)
            scores_archive[f"{name}_{r}"] = scores
        scores_archive["labels"] = labels
    np.savez_compressed(out/"classifier_scores.npz",**scores_archive)
    dump(out/"classifiers.json",classifier)
    return state_weights["kernel"]


def markov(out,weights):
    seq = quantize(weights)
    half = len(seq)//2
    train,test = seq[:half],seq[half:]
    phase,first = control_losses(train,test)
    hierarchical = DyadicMetaMarkov(feedback=False).fit(train)
    meta = DyadicMetaMarkov(feedback=True).fit(train)
    hl,ml = hierarchical.losses(test),meta.losses(test)
    values = {"phase_only":phase,"first_order":first,"dyadic_no_feedback":hl,"dyadic_meta":ml}
    result = {k:mean_ci(v) for k,v in values.items()}
    result["meta_minus_no_feedback"] = mean_ci(ml-hl)
    result["meta_minus_phase_only"] = mean_ci(ml-phase)
    result["training"] = {"train_sequences":half,"test_sequences":len(test),
        "bins":32,"tau":32,"cap":65535,"passes":1,
        "meta_contexts":len(meta.counts),"max_count":max(int(v.max()) for v in meta.counts.values())}
    witness = None
    for r in range(24):
        seen = {}
        for i,row in enumerate(seq):
            key = int(row[r])
            if key in seen:
                j = seen[key]
                if row[r+1] != seq[j,r+1]:
                    witness = {"phase":r,"sample_indices":[j,i],"same_bin":key,
                        "next_bins":[int(seq[j,r+1]),int(row[r+1])],
                        "current_weights":[int(weights[j,r]),int(weights[i,r])],
                        "next_weights":[int(weights[j,r+1]),int(weights[i,r+1])]}
                    break
            else:
                seen[key] = i
        if witness:
            break
    result["non_lumpability_witness"] = witness
    np.savez_compressed(out/"markov_losses.npz",**values)
    dump(out/"markov.json",result)
    print("finite meta-Markov evaluation complete",flush=True)


def causal_graphs(out,n):
    rng = np.random.default_rng(SEED+2)
    base = rng.integers(0,2,(n,25),dtype=np.uint64)
    base_out = base.copy()
    do0 = np.tile(base,(25,1,1))
    do1 = do0.copy()
    for i in range(25):
        do0[i,:,i] = 0
        do1[i,:,i] = 1
    arrays,summary = {},{}
    for r in range(1,13):
        base_out = round_batch(base_out,r-1,1)
        do0 = round_batch(do0,r-1,1)
        do1 = round_batch(do1,r-1,1)
        if r not in (1,2,3,4,6,12):
            continue
        influence = (do0 != do1).mean(axis=1)
        ate = (do1.astype(float)-do0).mean(axis=1)
        x,y = base.astype(float),base_out.astype(float)
        x -= x.mean(axis=0)
        y -= y.mean(axis=0)
        corr = (x.T@y)/np.sqrt(np.sum(x*x,axis=0)[:,None]*np.sum(y*y,axis=0)[None,:])
        arrays[f"influence_{r}"],arrays[f"ate_{r}"],arrays[f"correlation_{r}"] = influence,ate,corr
        scc = {}
        for threshold in (.1,.4,.6):
            graph = [np.flatnonzero(row >= threshold).tolist() for row in influence]
            comp = tarjan(graph)
            scc[str(threshold)] = {"components":len(comp),"largest":max(map(len,comp)),
                "edges":sum(map(len,graph))}
        summary[str(r)] = {"mean_influence":float(influence.mean()),
            "min_influence":float(influence.min()),"max_influence":float(influence.max()),
            "mean_abs_ate":float(np.abs(ate).mean()),"mean_abs_pearson":float(np.abs(corr).mean()),
            "threshold_scc":scc}
    structural = {}
    for w in (1,64):
        g = dependency_graph(w)
        comps = tarjan(g)
        structural[str(25*w)] = {"vertices":len(g),"edges":sum(map(len,g)),
            "components":len(comps),"largest":max(map(len,comps))}
    summary["structural"] = structural
    summary["settings"] = {"width":25,"contexts_per_intervention":n,
        "sampling":"uniform random 25-bit states, paired do(X_i=0/1)",
        "matrix_axes":"row=input bit, column=output bit"}
    np.savez_compressed(out/"causal_matrices.npz",**arrays)
    dump(out/"causal_graphs.json",summary)
    dump(out/"chi_2sat.json",chi_relation_analysis())
    print("causal interventions and Tarjan graphs complete",flush=True)


def cycle_summary(mapping):
    g = [[int(v)] for v in mapping]
    comps = tarjan(g)
    cyc = [c for c in comps if len(c)>1 or mapping[c[0]]==c[0]]
    return {"scc_count":len(comps),"cycles":len(cyc),
        "cyclic_vertices":sum(map(len,cyc)),"longest_cycle":max(map(len,cyc)),
        "cycle_lengths":sorted(map(len,cyc))}


def projected_cycles(out,k=12):
    n = 1 << k
    v = np.arange(n,dtype=np.uint64)
    state = np.zeros((n,25),dtype=np.uint64)
    state[:,:k] = (v[:,None] >> np.arange(k,dtype=np.uint64)) & 1
    results,arrays = {},{}
    for r in (1,2,12):
        dest = permute_batch(state,r,w=1)
        mapping = np.sum(dest[:,:k] << np.arange(k,dtype=np.uint64),axis=1).astype(np.int32)
        arrays[f"mapping_{r}"] = mapping
        results[str(r)] = cycle_summary(mapping)
    rng = np.random.default_rng(SEED+3)
    null = [cycle_summary(rng.integers(n,size=n)) for _ in range(64)]
    results["null"] = {key:mean_ci([d[key] for d in null])
                       for key in ("cycles","cyclic_vertices","longest_cycle")}
    results["definition"] = f"F(u) = first {k} bits of prefix-r Keccak-f[25](u || 0^(25-{k})); reinject at each step"
    results["state_count"] = n
    np.savez_compressed(out/"projected_maps.npz",**arrays)
    dump(out/"projected_cycles.json",results)
    dump(out/"projected_cycles_null.json",null)
    print("exhaustive projected cycles complete",flush=True)


def wilson(successes,n):
    p,z = successes/n,stats.norm.ppf(.975)
    center = (p+z*z/(2*n))/(1+z*z/n)
    radius = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return [float(center-radius),float(center+radius)]


def search_bench(out,trials,budget):
    rng = np.random.default_rng(SEED+4)
    rows,summary = [],{}
    for rounds in (2,3,24):
        by_method = {"random":[],"meta":[]}
        for trial in range(trials):
            target = int(rng.integers(4096))
            for j,method in enumerate(by_method):
                seed = SEED+100000*rounds+100*trial+j
                ans = truncated_preimage(target,rounds,seed,budget,12,method)
                ans.update({"rounds":rounds,"trial":trial,"target":target,"method":method,"seed":seed})
                rows.append(ans)
                by_method[method].append(ans)
            if (trial+1) % 8 == 0:
                print(f"12-bit search r={rounds}: {trial+1}/{trials}",flush=True)
        item = {}
        for method,rr in by_method.items():
            successes = sum(row["success"] for row in rr)
            item[method] = {"successes":successes,"trials":trials,
                "success_rate":successes/trials,"wilson_ci95":wilson(successes,trials),
                "mean_queries_censored":float(np.mean([row["queries"] for row in rr])),
                "total_queries":sum(row["queries"] for row in rr),
                "total_scc_checks":sum(row["scc_checks"] for row in rr),
                "total_forced_restarts":sum(row["forced_restarts"] for row in rr)}
        diffs = np.array([int(a["success"])-int(b["success"])
                for a,b in zip(by_method["meta"],by_method["random"])])
        item["paired_success_difference"] = mean_ci(diffs)
        discordant = np.count_nonzero(diffs)
        item["paired_exact_p"] = float(stats.binomtest(int(np.sum(diffs==1)),int(discordant),.5).pvalue) if discordant else 1.0
        summary[str(rounds)] = item
    summary["settings"] = {"budget":budget,"target_bits":12,"trials":trials,
        "ideal_random_success_probability":1-(1-2**-12)**budget,
        "query_accounting":"each distinct message evaluation counts; stop on hit; failures censored at budget"}
    dump(out/"search_trials.json",rows)
    dump(out/"search_summary.json",summary)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out",type=Path,default=Path("results"))
    parser.add_argument("--quick",action="store_true",help="smoke run; never use as paper data")
    args = parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    start = time.perf_counter()
    n,features_n,causal_n,trials,budget = ((256,128,128,4,128) if args.quick else (8192,2048,4096,48,2048))
    dump(args.out/"collision_certificate.json",collision_certificate())
    weights = avalanche(args.out,n,features_n)
    markov(args.out,weights)
    causal_graphs(args.out,causal_n)
    projected_cycles(args.out,8 if args.quick else 12)
    search_bench(args.out,trials,budget)
    code = {}
    for path in sorted(Path(__file__).parent.glob("*.py")):
        code[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {"seed":SEED,"quick":args.quick,"python":sys.version,"numpy":np.__version__,
        "scipy":__import__("scipy").__version__,"platform":platform.platform(),
        "elapsed_seconds":time.perf_counter()-start,"round_schedule":"prefix RC[0:r]",
        "message_length_bytes":128,"avalanche_pairs_per_difference":n,
        "classifier_examples_per_class":features_n,"causal_contexts":causal_n,
        "search_trials_per_method_per_round":trials,"search_budget":budget,
        "source_sha256":code}
    dump(args.out/"manifest.json",manifest)
    print(f"done in {manifest['elapsed_seconds']:.1f}s",flush=True)


if __name__ == "__main__":
    main()
