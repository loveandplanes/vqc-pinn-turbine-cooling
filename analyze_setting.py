"""analyze_setting.py — Is the benchmark setting itself rigging the game? (COPY only)

Checks (no training needed, uses cost table only):
 1. RAW (unnormalized) cost distribution — does [0,1] normalization flatten physics?
 2. Constant-policy baselines (always deploy idx 8/9/10/11) vs oracle tracker,
    in RAW units — if a no-learning constant policy ties the trained agents,
    the setting tests 'ability to collapse', not learning.
"""
import torch
import torch.nn.functional as F
import sys
sys.path.insert(0, ".")
from rigorous_benchmark import precompute_frozen_codebook
from turbine_pinn_geometry import CoolingChannelPINN, N_CH, N_H
import warnings
warnings.filterwarnings("ignore")

N, NS = 5, 32
pinn = CoolingChannelPINN(input_dim=NS, hidden_dim=128)
fcb = precompute_frozen_codebook(NS, pinn)

# RAW costs (same loop as precompute_codebook_costs, WITHOUT normalization)
pressures = torch.linspace(0.5, 1.2, 16)
raw = torch.zeros(16, NS)
with torch.no_grad():
    for p_idx, pressure in enumerate(pressures):
        cfd_row = torch.tensor([[340.0, pressure.item(), 10.0, 2.25]])
        for i in range(NS):
            dp = fcb[i:i + 1]
            ch = dp[:, :N_CH * 3].reshape(-1, N_CH, 3)
            h = dp[:, N_CH * 3:].reshape(-1, N_H, 2)
            xc = torch.sigmoid(ch[:, :, 0]) * 0.74 + 0.06
            yc = torch.tanh(ch[:, :, 1]) * 0.055 + 0.02
            Rc = F.softplus(ch[:, :, 2]) * 0.010 + 0.006
            sh = torch.sigmoid(h[:, :, 0])
            dh = F.softplus(h[:, :, 1]) * 0.003 + 0.001
            raw[p_idx, i] = pinn.calculate_cost_signal(xc, yc, Rc, sh, dh, cfd_row).item()

print("=" * 70)
print("RAW cost table stats (physics units, no normalization):")
print(f"  min={raw.min():.6f}  median={raw.median():.6f}  max={raw.max():.2f}")
print(f"  max/min ratio = {raw.max() / (raw.min() + 1e-12):.2e}")
print(f"  per-pressure best idx: {raw.argmin(dim=1).tolist()}")

mean_raw = raw.mean(dim=0)
order = torch.argsort(mean_raw)
print(f"\n  designs by mean RAW cost (top 8): {order[:8].tolist()}")
print(f"  their means: {[f'{mean_raw[i]:.4f}' for i in order[:8]]}")

oracle_raw = raw.min(dim=1).values.mean().item()
print(f"\n  CONSTANT POLICY mean RAW cost (deploy same idx everywhere):")
for idx in [8, 9, 10, 11]:
    print(f"    always-{idx}: {raw[:, idx].mean():.6f}")
print(f"  ORACLE tracker (best idx per pressure): {oracle_raw:.6f}")
print(f"  constant-10 penalty vs oracle: {mean_raw[10].item() - oracle_raw:.6f} "
      f"({(mean_raw[10].item() / (oracle_raw + 1e-12) - 1) * 100:.1f}% worse)")

# Same comparison in NORMALIZED units (what training actually sees)
cmin, cmax = raw.min(), raw.max()
norm = (raw - cmin) / (cmax - cmin + 1e-8)
print(f"\nNORMALIZED units (training signal): cmin={cmin:.4f} cmax={cmax:.2f}")
print(f"  constant-10 mean: {norm[:, 10].mean():.6f}  oracle: {norm.min(dim=1).values.mean():.6f}")
print(f"  gap in normalized units: {norm[:, 10].mean().item() - norm.min(dim=1).values.mean().item():.6f}")
print(f"  -> gradient scale for 'track context instead of collapse' is ~this gap.")
print("=" * 70)
