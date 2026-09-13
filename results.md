# Results (consolidated, honest)

Supersedes the earlier `benchmark_report.md` and `quantum_advantage_analysis.md`
(removed; preserved in git history). Those notes predate the rescaled benchmark
and contain claims — "tunneling", +23% figures on the old cost scale — that the
current results do not support. What follows is only what was measured.

Setup for everything below: N=5 qubits, 32-design frozen codebook, shared
log+per-row-rescaled cost table (16 pressures, 0.5–1.2 atm), Adam 0.02 + cosine
schedule, forced final anneal (last 50 epochs, T→0.05/τ→0.1). Quantum: 61 params.
Small classical: 336. Big classical control: 6,624.

## Flagship: 500 epochs, seed 42

| agent | train E-cost | fair deployed | gradients | designs tracked |
|---|---|---|---|---|
| Quantum self-reg | 0.031 | **0.108** | alive all 500 epochs | 4 |
| Quantum fixed T=0.01 | 0.031 | 0.153 | flickers, converges anyway | 9 |
| Classical self-reg | 0.377 | 0.292 | dead (0.00) from ~epoch 50 | 1 |
| Classical fixed | 0.377 | 0.303 | dead from epoch 50 | 1 |

Notes: classical self-reg and fixed trajectories are byte-identical at every
epoch — once collapsed to a delta, temperature is irrelevant. Constant-10
(no-learning) policy scores 0.307 on this table: classical never beat doing nothing.

## Repeats: 150 epochs, 4 seeds (quantum vs small classical, fair deployed)

| seed | Q train / deployed / div | C train / deployed / div |
|---|---|---|
| 42 | 0.174 / **0.293** / 5 | 0.199 / 0.309 / 1 |
| 123 | 0.344 / **0.220** / 6 | 0.229 / 0.306 / 1 |
| 7 | 0.216 / **0.261** / 6 | 0.279 / 0.281 / 1 |
| 2026 | 0.150 / **0.177** / 5 | 0.293 / 0.299 / 1 |

Quantum wins deployed 4/4, train-E 3/4. Seed 123: trails on train, wins deployed.

## Control: big classical (6,624 params, ~108x quantum), 150 epochs

| seed | train | fair deployed | div | grads |
|---|---|---|---|---|
| 42 | 0.228 | 0.330 | 1 | dead from epoch 50 |
| 123 | 0.424 | 0.311 | 1 | dead from epoch 50 |

Worst of all six configurations. More capacity → faster saturation → earlier
death. Collapse is dynamical, not a capacity shortage.

## Ablations behind the fixes

- Cost scaling shootout (gap constant-10 vs oracle): global min-max 0.016,
  global-log 0.246, per-row-log **0.307**, ranks 0.034, winsorized 0.017.
- Forced anneal on hard seed: train-E 0.35 → 0.025.
- Learnable quantum logit scale: trained to 1.00 in every run (parity
  insurance; effectively pure log mapping in all reported numbers).

## Caveats (read before citing)

- Flagship run is single-seed; robustness rests on the four 150-epoch repeats.
- Fair deployed trails train deployed (fresh-context generalization gap).
- Oracle (0.00) unreached by all agents — the tracking task is genuinely hard.
- N=5/32-menu scale; surrogate (not full-CFD) physics; one-dimensional
  (pressure-only) context variation.
- Claims: collapse-resistant exploration, parameter efficiency, gradient
  health. Not supremacy. Mechanism: reversible (temperature) vs irreversible
  (weight-saturation) sharpening — no tunneling events observed or claimed.
