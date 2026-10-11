# Experimental definitions and scope

## Targets

`H^(r)` keeps SHA3-256's 1088-bit rate, 512-bit capacity, `0x06` suffix and standard padding. Every permutation call uses prefix rounds `0,...,r-1`. The primary inputs are 128 bytes, so each uses one padded rate block. `r=24` is standard SHA3-256. This is distinct from Keccak-p's suffix-round naming convention.

All results are reproducible from seed **20261011** and the offsets in source. The protocol was implemented before the primary run; it was not externally preregistered. The independent replication was added after observing the primary two-round search result. No algorithm parameters were tuned on replication outcomes.

## Three proposed analysis routes

1. **Hypercomplex:** 25 real Clifford multivectors in `Cl(0,6)`, each with 64 spin coefficients. Coordinatewise Boolean operations and the noncommutative geometric product are distinct. Transported rounds are exact. The coefficient squared distance is `4*Hamming`; seven product-grade energies are tested as additional features.
2. **Finite self-referential Markov:** finite dyadic refinements and feedback derived from the predictor's own preceding modal prediction. This gives a precise interpretation of the proposed finite fractal architecture. No premise of infinite fractality or exact Markovianity of a lossy projection is used.
3. **Tarjan, 2-SAT and causality:** separate graphs for coordinate dependencies, finite projected dynamics, implication constraints and intervention influence. Signed average effects and Pearson correlation are recorded alongside paired flip influence. A majority-closure witness proves the local nonlinear χ relation is not exactly expressible by existential 2-CNF.

## Fixed choices

| Experiment | Settings |
|---|---|
| Diffusion | 8,192 messages; two fixed weight-two differences; all rounds 0–24 recorded. |
| Kernel difference | Lane 5 bit 0 and lane 10 bit 0; message bytes 40 and 80, low bit. |
| Control difference | Lane 0 bit 0 and lane 1 bit 0; message bytes 0 and 8, low bit. |
| Classification | 2,048 examples per class; half train/half test; ridge penalty 1; fixed grade pairs; 300 conditional bootstrap resamples. |
| Trajectory predictor | 32 output bins; six nested input-bin levels; phase included; smoothing weight 32; cap 65,535; one pass; 4,096 train and 4,096 test sequences. |
| Influence | Width 25; 4,096 uniform contexts per intervened bit; checkpoints 1,2,3,4,6,12. |
| Structural SCC | Widths 25 and 1,600; input and output coordinate labels identified. |
| Projected cycles | All 4,096 states; inject 12 bits into a zero-filled 25-bit state, permute, project, reinject; rounds 1,2,12; 64 random-map controls. |
| Primary search | 12 target bits; 2,048 distinct-message query cap; 48 targets at 2,3,24 rounds; random/full model. |
| Search policy | Actions: flip 1,2,4,8 of 1,024 input bits or fresh message; epsilon .25; accept equal/better or .03 exploration; finite counts capped at 255. |
| Search feedback | Correctness of the previous predicted modal distance; nested contexts `h >> s`, `s=4,3,2,1,0`; smoothing weight 16; binomial prior. |
| Search SCC | Distances 0–12 as vertices; accepted moves or rejected self-loops; check every 16 queries after 32 without a new global best. |
| Replication | New seeds/targets; 48 targets at 2 and 24 rounds; random, hill climb, no feedback, no SCC, full model. |

## Statistical interpretation

- Message/trajectory is the unit for mean intervals; correlated bits do not count as independent messages.
- Classifier bootstrap intervals condition on the learned score and omit training-set uncertainty. AUC orientation is fixed using training data; test scores below 0.5 are retained.
- Digest-bit binomial tests at 24 rounds form one 512-test Holm family.
- The primary search uses exact paired success tests with Holm correction across three round counts. Follow-up contrasts are explicitly reported as follow-up; the paper also states their familywise interpretation.
- A confidence interval for a random-map *mean* is not the expected range of one random map.
- Truncated-target hits and fewer hash queries do not establish full-output preimages or wall-clock acceleration.
- The source fingerprints in `results/manifest.json` bind the recorded primary data to the final code. `scripts/verify_results.py` checks them and verifies all 768 stored search endpoints, the reduced collision, raw-array means, and projected cycles using independent indegree pruning.

## Files and exact replay

JSON stores configuration, measured summaries and complete endpoint messages. NPZ files store the full weight trajectories and numerical diagnostic arrays. Input messages are regenerated from the seed, shape, dtype and sampling calls in `experiments.py`; classifier null samples and bootstrap streams likewise follow the saved code. Re-running reproduces scientific values; elapsed wall times and compressed archive metadata can differ.

`SHA256SUMS` records the release file bytes. Regenerating a figure, a PDF or an NPZ archive may change file hashes without changing the numerical experiment. Run the result verifier to check the scientific records themselves.
