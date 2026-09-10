import torch
import torch.nn.functional as F
from agent_1_quantum_gumbel import QuantumArchitectGumbel
from turbine_pinn_geometry import CoolingChannelPINN, N_CH, N_H

agent = QuantumArchitectGumbel(n_qubits=8, n_layers=5)
# Use the same frozen codebook strategy
agent.codebook.requires_grad_(False)

env_ctx = torch.randn(2, 8)
cfd = torch.randn(2, 4)

design_params, q_probs = agent(env_ctx)

ch = design_params[:, :N_CH*3].reshape(-1, N_CH, 3)
h  = design_params[:, N_CH*3:].reshape(-1, N_H,  2)
xc = torch.sigmoid(ch[:, :, 0]) * 0.74 + 0.06
yc = torch.tanh   (ch[:, :, 1]) * 0.055 + 0.02
Rc = F.softplus   (ch[:, :, 2]) * 0.010 + 0.006
sh = torch.sigmoid(h[:, :, 0])
dh = F.softplus   (h[:, :, 1]) * 0.003 + 0.001

pinn = CoolingChannelPINN()
cost = pinn.calculate_cost_signal(xc, yc, Rc, sh, dh, cfd)

cost.backward()

total_norm = 0.0
has_grad = False
for name, p in agent.named_parameters():
    if p.requires_grad:
        if p.grad is not None:
            norm = p.grad.norm().item()
            print(f"{name}: {norm}")
            total_norm += norm
            has_grad = True
        else:
            print(f"{name}: None gradient")
print(f"Total grad norm: {total_norm}")
