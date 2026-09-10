"""verify_big_classical.py — Can a BIG classical MLP learn tracking? (COPY only)

Control arm: ClassicalArchitectGumbel with matched_capacity=False, hidden=64
(~6.6k params, ~100x the quantum agent) vs the 61-param quantum agent.
Same table, same optimizer/schedule/anneal, seeds 42 + 123.
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

big = ClassicalArchitectGumbel(n_features=N, n_states=NS, hidden_dim=64, matched_capacity=False)
n_big = sum(p.numel() for p in big.prob_net.parameters())
print(f"BIG classical prob_net params: {n_big} (vs quantum 61, small classical 336)")


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
    print(f"--- seed {seed} ---", flush=True)
    torch.manual_seed(seed)
    ca = ClassicalArchitectGumbel(n_features=N, n_states=NS, hidden_dim=64, matched_capacity=False)
    cpinn = CoolingChannelPINN(input_dim=NS, hidden_dim=128)
    hc = run_frozen_codebook_agent(ca, cpinn, fcb.clone(), N, EPOCHS, is_dynamic=True,
                                   cost_table=ct, cost_pressures=cp, final_anneal_epochs=50)
    cd, cdiv = fair_eval(ca)
    print(f"  BIG-C: trainE={hc['loss'][-1]:.4f} fairDep={cd:.4f} div={cdiv} Tend={hc['temp'][-1]:.3f}",
          flush=True)
print("Reference (150ep, same setup): Q s42 trainE=0.174 fair=0.293 div=5 | "
      "Q s123 trainE=0.344 fair=0.220 div=6", flush=True)
print("CONTROL DONE", flush=True)
