"""verify_rescaled.py — Does log+per-row scaling fix the benchmark? (COPY only)

Trains Quantum vs Classical (self-regulating, forced anneal on) on the NEW
log_row cost table. Honest outcomes only — prints whichever wins.
"""
import torch
import sys
sys.path.insert(0, ".")
from agent_1_quantum_gumbel import QuantumArchitectGumbel, ClassicalArchitectGumbel
from turbine_pinn_geometry import CoolingChannelPINN
from rigorous_benchmark import precompute_frozen_codebook, precompute_codebook_costs, run_frozen_codebook_agent
from data_loader import get_cfd_dataloader
import numpy as np
import warnings
warnings.filterwarnings("ignore")

N, NS, EPOCHS = 5, 32, 150
pinn0 = CoolingChannelPINN(input_dim=NS, hidden_dim=128)
fcb = precompute_frozen_codebook(NS, pinn0)
ct, cp = precompute_codebook_costs(fcb, pinn0, n_cfd_samples=16, transform="log_row")
print(f"  Oracle(row-best) mean on new table: {ct.min(dim=1).values.mean():.4f} (=0 by construction)")
print(f"  Constant-10 mean on new table:      {ct[:, 10].mean():.4f} (collapse now LOSES)")


def fair_eval(agent, n_batches=40):
    agent.temperature, agent.gumbel_tau = 0.01, 0.1
    dl = get_cfd_dataloader(batch_size=8, samples=n_batches * 8, n_features=N)
    deps, picks = [], []
    agent.eval()
    with torch.no_grad():
        for env_ctx, cfd in dl:
            _, probs = agent(env_ctx)
            idx = probs.argmax(dim=-1)
            pv = cfd[:, 1].clamp(0.5, 1.2)
            pi = torch.argmin((cp.unsqueeze(0) - pv.unsqueeze(1)).abs(), dim=1)
            costs = ct[pi]
            deps.append(costs[torch.arange(len(idx)), idx].mean().item())
            picks.extend(idx.tolist())
    return float(np.mean(deps)), len(set(picks))


for seed in [42, 123]:
    print(f"\n--- seed {seed} ---")
    torch.manual_seed(seed)
    qa = QuantumArchitectGumbel(n_qubits=N, n_layers=4)
    qp = CoolingChannelPINN(input_dim=NS, hidden_dim=128)
    hq = run_frozen_codebook_agent(qa, qp, fcb.clone(), N, EPOCHS, is_dynamic=True,
                                   cost_table=ct, cost_pressures=cp, final_anneal_epochs=50)
    torch.manual_seed(seed)
    ca = ClassicalArchitectGumbel(n_features=N, n_states=NS, hidden_dim=8, matched_capacity=True)
    cpinn = CoolingChannelPINN(input_dim=NS, hidden_dim=128)
    hc = run_frozen_codebook_agent(ca, cpinn, fcb.clone(), N, EPOCHS, is_dynamic=True,
                                   cost_table=ct, cost_pressures=cp, final_anneal_epochs=50)
    qd, qdiv = fair_eval(qa)
    cd, cdiv = fair_eval(ca)
    print(f"  Q: trainE={hq['loss'][-1]:.4f} fairDep={qd:.4f} div={qdiv} "
          f"Tend={hq['temp'][-1]:.3f} scale={float(torch.clamp(qa.logit_scale, 1.0, 30.0)):.2f}")
    print(f"  C: trainE={hc['loss'][-1]:.4f} fairDep={cd:.4f} div={cdiv} Tend={hc['temp'][-1]:.3f}")
