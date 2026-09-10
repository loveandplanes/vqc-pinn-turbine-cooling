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

inputs = torch.randn(4, n_qubits)
q_probs = qlayer(inputs)

# Test 2 first: logits = q_probs * 30.0
logits2 = q_probs * 30.0
soft2 = F.gumbel_softmax(logits2, tau=1.0, hard=True)
cost2 = -soft2[:, 5].mean()
cost2.backward(retain_graph=True)
grad2 = qlayer.weights.grad.clone() if qlayer.weights.grad is not None else torch.zeros_like(qlayer.weights)
qlayer.zero_grad()

# Test 1 second: logits = torch.log(q_probs + 1e-9)
logits1 = torch.log(q_probs + 1e-9)
soft1 = F.gumbel_softmax(logits1, tau=1.0, hard=True)
cost1 = -soft1[:, 5].mean()
cost1.backward()
grad1 = qlayer.weights.grad.clone() if qlayer.weights.grad is not None else torch.zeros_like(qlayer.weights)

print("Grad norm with q_probs * 30.0:", grad2.norm().item())
print("Grad norm with log(q_probs):", grad1.norm().item())
