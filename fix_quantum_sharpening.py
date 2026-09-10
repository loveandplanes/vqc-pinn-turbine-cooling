"""
fix_quantum_sharpening.py — Verify the fair-sharpening fix (COPY folder only).

Compares, with identical training (expected cost, frozen codebook, N=5/32):
  1. Classical Gumbel (orig, unbounded logits)
  2. Quantum Gumbel PATCHED (learnable logit_scale, init 1.0)

Reports BOTH metrics:
  - Expected cost  E[cost] = probs @ costs   (rewards collapse)
  - Deployed cost  cost(argmax design)       (what you actually manufacture)

Fair final eval is done at fixed low T=0.01 for both agents.
Run:  python fix_quantum_sharpening.py   (~3-6 min on CPU)
"""
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import sys, time
sys.path.insert(0, ".")

from agent_1_quantum_gumbel import QuantumArchitectGumbel, ClassicalArchitectGumbel
from turbine_pinn_geometry import CoolingChannelPINN
from rigorous_benchmark import (
    precompute_frozen_codebook, precompute_codebook_costs,
    run_frozen_codebook_agent,
)
from data_loader import get_cfd_dataloader

EPOCHS = 150
N = 5
N_STATES = 2 ** N
C_HIDDEN = 8

print("=" * 70)
print("VERIFY FIX: Quantum learnable logit_scale + deployed-cost evaluation")
print(f"N={N} ({N_STATES} states) | {EPOCHS} epochs each | frozen codebook")
print("=" * 70)

q0 = QuantumArchitectGumbel(n_qubits=N, n_layers=4)
c0 = ClassicalArchitectGumbel(n_features=N, n_states=N_STATES, hidden_dim=C_HIDDEN, matched_capacity=True)
print(f"Quantum params (circuit+scale): {sum(p.numel() for p in q0.qlayer.parameters()) + 1} "
      f"(circuit {sum(p.numel() for p in q0.qlayer.parameters())} + scale 1)")
print(f"Classical params (prob_net):    {sum(p.numel() for p in c0.prob_net.parameters())}")

dummy_pinn = CoolingChannelPINN(input_dim=N_STATES, hidden_dim=128)
frozen_cb = precompute_frozen_codebook(N_STATES, dummy_pinn)
cost_table, cost_pressures = precompute_codebook_costs(frozen_cb, dummy_pinn, n_cfd_samples=16)

t0 = time.time()

print("\n[1/2] Quantum PATCHED (learnable logit_scale, self-regulating)...")
torch.manual_seed(42)
q_agent = QuantumArchitectGumbel(n_qubits=N, n_layers=4)
q_pinn = CoolingChannelPINN(input_dim=N_STATES, hidden_dim=128)
hist_q = run_frozen_codebook_agent(q_agent, q_pinn, frozen_cb.clone(), N, EPOCHS,
                                   is_dynamic=True, cost_table=cost_table,
                                   cost_pressures=cost_pressures)
print(f"  train E[cost] final: {hist_q['loss'][-1]:.4f} | "
      f"train deployed final: {hist_q['discrete_cost'][-1]:.4f} | "
      f"learned scale: {torch.clamp(q_agent.logit_scale, 1.0, 30.0).item():.2f}")

print("\n[2/2] Classical (self-regulating)...")
torch.manual_seed(42)
c_agent = ClassicalArchitectGumbel(n_features=N, n_states=N_STATES, hidden_dim=C_HIDDEN, matched_capacity=True)
c_pinn = CoolingChannelPINN(input_dim=N_STATES, hidden_dim=128)
hist_c = run_frozen_codebook_agent(c_agent, c_pinn, frozen_cb.clone(), N, EPOCHS,
                                   is_dynamic=True, cost_table=cost_table,
                                   cost_pressures=cost_pressures)
print(f"  train E[cost] final: {hist_c['loss'][-1]:.4f} | "
      f"train deployed final: {hist_c['discrete_cost'][-1]:.4f}")

print(f"\nTraining done in {(time.time()-t0)/60:.1f} min")


def fair_eval(agent, n_batches=50, batch_size=8):
    """Deployed-cost eval at fixed low T=0.01 (fair sharpening for both)."""
    agent.temperature = 0.01
    agent.gumbel_tau = 0.1
    dl = get_cfd_dataloader(batch_size=batch_size, samples=n_batches * batch_size, n_features=N)
    deployed, entropies, picks = [], [], []
    agent.eval()
    with torch.no_grad():
        for env_ctx, cfd in dl:
            _, probs = agent(env_ctx)
            idx = probs.argmax(dim=-1)
            pvals = cfd[:, 1].clamp(0.5, 1.2)
            p_idx = torch.argmin((cost_pressures.unsqueeze(0) - pvals.unsqueeze(1)).abs(), dim=1)
            costs = cost_table[p_idx]
            deployed.append(costs[torch.arange(len(idx)), idx].mean().item())
            entropies.append((-(probs * torch.log(probs + 1e-9)).sum(dim=-1)).mean().item())
            picks.extend(idx.tolist())
    return float(np.mean(deployed)), float(np.mean(entropies)), len(set(picks)), picks


q_dep, q_ent, q_div, q_picks = fair_eval(q_agent)
c_dep, c_ent, c_div, c_picks = fair_eval(c_agent)

print("\n" + "=" * 70)
print("FAIR EVAL @ T=0.01 (what you would actually manufacture):")
print(f"  Quantum PATCHED: deployed={q_dep:.4f} entropy={q_ent:.2f} unique={q_div}")
print(f"  Classical:       deployed={c_dep:.4f} entropy={c_ent:.2f} unique={c_div}")
print(f"  Train E[cost] final: Q={hist_q['loss'][-1]:.4f} vs C={hist_c['loss'][-1]:.4f}")
print(f"  Train deployed final: Q={hist_q['discrete_cost'][-1]:.4f} vs C={hist_c['discrete_cost'][-1]:.4f}")
from collections import Counter
print(f"  Q top picks: {Counter(q_picks).most_common(5)}")
print(f"  C top picks: {Counter(c_picks).most_common(5)}")
print("=" * 70)

# Plot: expected vs deployed
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle(f'Fix verification (COPY): Expected vs Deployed cost | N={N} | {EPOCHS} epochs',
             fontsize=12)
from scipy.ndimage import uniform_filter1d
smooth = lambda x, w=9: uniform_filter1d(np.array(x, dtype=float), size=w)
x = np.arange(1, EPOCHS + 1)
ax = axes[0]
ax.semilogy(x, smooth(hist_q['loss']), color='#00d4ff', lw=2, label='Quantum patched E[cost]')
ax.semilogy(x, smooth(hist_c['loss']), color='#2ecc71', lw=2, label='Classical E[cost]')
ax.set_title('Training: Expected cost (smooth feeds grads)'); ax.set_xlabel('Epoch')
ax.legend(fontsize=8); ax.grid(True, ls=':', alpha=0.3)
ax = axes[1]
ax.semilogy(x, smooth(hist_q['discrete_cost']), color='#00d4ff', lw=2, label='Quantum patched deployed')
ax.semilogy(x, smooth(hist_c['discrete_cost']), color='#2ecc71', lw=2, label='Classical deployed')
ax.set_title('Training: Deployed argmax cost (fair metric)'); ax.set_xlabel('Epoch')
ax.legend(fontsize=8); ax.grid(True, ls=':', alpha=0.3)
ax = axes[2]
ax.bar(['Q patched\ndeployed@T=0.01', 'Classical\ndeployed@T=0.01'], [q_dep, c_dep],
       color=['#00d4ff', '#2ecc71'], edgecolor='white')
ax.set_title('Fair final eval: deployed cost (lower=better)')
for i, v in enumerate([q_dep, c_dep]):
    ax.text(i, v, f'{v:.4f}', ha='center', va='bottom', fontsize=10)
plt.tight_layout()
plt.savefig('fix_verification.png', dpi=200, bbox_inches='tight')
print("Saved: fix_verification.png")
