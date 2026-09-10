import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import pennylane as qml

n_qubits = 4
n_states = 16
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev, interface="torch")
def circuit(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(n_qubits))
    qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
    return qml.probs(wires=range(n_qubits))

weight_shapes = {"weights": (3, n_qubits, 3)}
qlayer = qml.qnn.TorchLayer(circuit, weight_shapes)
for p in qlayer.parameters():
    nn.init.normal_(p, mean=0.0, std=0.02)

optimizer = optim.Adam(qlayer.parameters(), lr=0.05) # larger lr

# Target: always select index 5
target_idx = 5

for epoch in range(100):
    optimizer.zero_grad()
    inputs = torch.randn(4, n_qubits) # batch size 4
    
    q_probs = qlayer(inputs)
    logits = torch.log(q_probs + 1e-9)
    soft_onehot = F.gumbel_softmax(logits, tau=1.0, hard=True)
    
    cost = -soft_onehot[:, target_idx].mean()
    
    cost.backward()
    optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        entropy = -(q_probs * torch.log(q_probs + 1e-9)).sum(dim=-1).mean().item()
        print(f"Epoch {epoch+1:2d} | Cost: {cost.item():.4f} | Prob of 5: {q_probs[:, target_idx].mean().item():.4f} | Entropy: {entropy:.4f}")
