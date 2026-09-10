import torch, sys
sys.path.insert(0, ".")
from agent_1_quantum_gumbel import QuantumArchitectGumbel, ClassicalArchitectGumbel
from turbine_pinn_geometry import CoolingChannelPINN
from rigorous_benchmark import precompute_frozen_codebook, precompute_codebook_costs, run_frozen_codebook_agent
import warnings
warnings.filterwarnings("ignore")

N = 5
NS = 32
EPOCHS = 80
pinn0 = CoolingChannelPINN(input_dim=NS, hidden_dim=128)
fcb = precompute_frozen_codebook(NS, pinn0)
ct, cp = precompute_codebook_costs(fcb, pinn0, n_cfd_samples=16)
for seed in [123, 999]:
    torch.manual_seed(seed)
    qa = QuantumArchitectGumbel(n_qubits=N, n_layers=4)
    qp = CoolingChannelPINN(input_dim=NS, hidden_dim=128)
    hq = run_frozen_codebook_agent(qa, qp, fcb.clone(), N, EPOCHS, is_dynamic=True,
                                   cost_table=ct, cost_pressures=cp)
    torch.manual_seed(seed)
    ca = ClassicalArchitectGumbel(n_features=N, n_states=NS, hidden_dim=8, matched_capacity=True)
    cpinn = CoolingChannelPINN(input_dim=NS, hidden_dim=128)
    hc = run_frozen_codebook_agent(ca, cpinn, fcb.clone(), N, EPOCHS, is_dynamic=True,
                                   cost_table=ct, cost_pressures=cp)
    qscale = float(torch.clamp(qa.logit_scale, 1.0, 30.0))
    print(f"seed {seed}: Q E={hq['loss'][-1]:.4f} dep={hq['discrete_cost'][-1]:.4f} "
          f"scale={qscale:.2f} | C E={hc['loss'][-1]:.4f} dep={hc['discrete_cost'][-1]:.4f}",
          flush=True)
