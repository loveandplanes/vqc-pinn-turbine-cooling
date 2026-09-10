"""verify_anneal.py — Does forced final anneal let quantum converge? (COPY only)."""
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

N, NS, EPOCHS, ANNEAL = 5, 32, 150, 50
pinn0 = CoolingChannelPINN(input_dim=NS, hidden_dim=128)
fcb = precompute_frozen_codebook(NS, pinn0)
ct, cp = precompute_codebook_costs(fcb, pinn0, n_cfd_samples=16)


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


def run(kind, seed, anneal):
    torch.manual_seed(seed)
    if kind == "Q":
        ag = QuantumArchitectGumbel(n_qubits=N, n_layers=4)
    else:
        ag = ClassicalArchitectGumbel(n_features=N, n_states=NS, hidden_dim=8, matched_capacity=True)
    pinn = CoolingChannelPINN(input_dim=NS, hidden_dim=128)
    h = run_frozen_codebook_agent(ag, pinn, fcb.clone(), N, EPOCHS, is_dynamic=True,
                                  cost_table=ct, cost_pressures=cp,
                                  final_anneal_epochs=(ANNEAL if anneal else 0))
    dep, div = fair_eval(ag)
    scale = ""
    if kind == "Q":
        scale = f" scale={float(torch.clamp(ag.logit_scale, 1.0, 30.0)):.2f}"
    tag = "anneal" if anneal else "no-anneal"
    print(f"  {kind} {tag} seed={seed}: trainE={h['loss'][-1]:.4f} "
          f"trainDep={h['discrete_cost'][-1]:.4f} fairDep={dep:.4f} div={div} "
          f"Tend={h['temp'][-1]:.3f}{scale}", flush=True)
    return dep


print("seed 42 (easy, annealed before):")
run("Q", 42, False)
run("Q", 42, True)
run("C", 42, True)
print("seed 123 (hard, quantum previously stuck at E=0.30):")
run("Q", 123, False)
run("Q", 123, True)
run("C", 123, True)
