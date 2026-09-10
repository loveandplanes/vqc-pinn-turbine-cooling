# Test audit (`test_*.py`)

Early circuit probes from the exploratory phase. Honest status: two are
well-designed, the rest are lab notes. Evidentiary weight for the project's
claims sits in `verify_*.py` + the 500-epoch `rigorous_benchmark.py` run,
not here.

| test | well-designed? | proves |
|---|---|---|
| `q_init` | ✅ yes | init fix (near-identity init, used in agent) |
| `hard_vs_soft` | ✅ yes | soft-train / hard-eval (used in design) |
| `q_grad` | ⚠️ smoke only | grads exist at N=8, single batch, unphysical CFD row |
| `scale` / `log` pair | ❌ confounded | nothing (lr 0.01 vs 0.05 mismatch + `hard=True` bias) |
| `grad_comparison` | ❌ no | nothing (single biased grad-norm; norm ≠ learnability) |
| `grad_flow` | ⚠️ probe only | gradient flows; not which mapping wins |

Notes:
- `q_init` and `hard_vs_soft` are matched A/Bs with isolated variables;
  both fixes they motivated are in the shipped agent.
- The scale-vs-log cluster is confounded three ways (learning-rate mismatch,
  straight-through bias, unjustified magic constant 30) and was superseded
  when training moved to expected cost (no Gumbel in the training loop).
- Shipped mapping is `log(q_probs)` × learnable scale (init 1.0; trained to
  1.00 in all runs, i.e. effectively pure log). The `×30` variant was tested,
  never adopted.
