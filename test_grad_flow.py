import torch
import torch.nn.functional as F
import torch.nn as nn
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

# TEST 1: log(q_probs)
q_probs1 = qlayer(inputs)
q_probs1.retain_grad()
logits1 = torch.log(q_probs1 + 1e-9)
logits1.retain_grad()
soft1 = F.gumbel_softmax(logits1, tau=1.0, hard=True)
cost1 = -soft1[:, 5].mean()
cost1.backward()

print("=== LOG METHOD ===")
print("logits grad norm:", logits1.grad.norm().item())
print("q_probs grad norm:", q_probs1.grad.norm().item())

qlayer.zero_grad()

# TEST 2: q_probs * 30
q_probs2 = qlayer(inputs)
q_probs2.retain_grad()
logits2 = q_probs2 * 30.0
logits2.retain_grad()
soft2 = F.gumbel_softmax(logits2, tau=1.0, hard=True)
cost2 = -soft2[:, 5].mean()
cost2.backward()

print("=== SCALE METHOD ===")
print("logits grad norm:", logits2.grad.norm().item())
print("q_probs grad norm:", q_probs2.grad.norm().item())

