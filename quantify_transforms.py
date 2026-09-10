"""quantify_transforms.py — Which cost scaling restores the tracking signal? (COPY only)

Compares candidate cost-table transforms by one number: the normalized gap
between 'always deploy idx 10' (classical's collapse strategy) and the
per-pressure oracle (quantum's tracking strategy). Bigger gap = stronger,
learnable gradient signal for doing the RIGHT thing.
"""
import torch
import sys
sys.path.insert(0, ".")
import warnings
warnings.filterwarnings("ignore")
from analyze_setting import raw  # noqa: reuses already-built RAW table

oracle_idx = raw.argmin(dim=1)

def gap_of(t):
    """Mean over pressures of (t[always-10] - t[oracle])."""
    return (t[torch.arange(t.shape[0]), 10] - t[torch.arange(t.shape[0]), oracle_idx]).mean().item()

print("\n" + "=" * 70)
print("TRANSFORM SHOOTOUT — signal gap 'constant-10 vs oracle' (bigger=better):")

# (a) current: global min-max on raw
cmin, cmax = raw.min(), raw.max()
t_a = (raw - cmin) / (cmax - cmin + 1e-8)
print(f"  (a) global min-max (CURRENT):            gap={gap_of(t_a):.6f}")

# (b) global min-max on log(raw)
lraw = torch.log(raw + 1.0)
t_b = (lraw - lraw.min()) / (lraw.max() - lraw.min() + 1e-8)
print(f"  (b) global min-max on log(cost):         gap={gap_of(t_b):.6f}")

# (c) per-pressure-row min-max on log(cost): best=0/worst=1 in every row
rmin = lraw.min(dim=1, keepdim=True).values
rmax = lraw.max(dim=1, keepdim=True).values
t_c = (lraw - rmin) / (rmax - rmin + 1e-8)
print(f"  (c) per-row min-max on log(cost):        gap={gap_of(t_c):.6f}")

# (d) rank per row / N_STATES (fully outlier-proof; oracle=0 by construction)
ranks = raw.argsort(dim=1).argsort(dim=1).float() / (raw.shape[1] - 1)
print(f"  (d) per-row rank/N (oracle=0):           gap={gap_of(ranks):.6f}")

# (e) winsorize raw at p99, then global min-max
cap = torch.quantile(raw.flatten(), 0.99)
wraw = torch.clamp(raw, max=cap)
t_e = (wraw - wraw.min()) / (wraw.max() - wraw.min() + 1e-8)
print(f"  (e) winsorize p99 (cap={cap:.0f}) + min-max: gap={gap_of(t_e):.6f}")

print("=" * 70)
