import torch
import torch.nn as nn
import pennylane as qml

n_qubits = 8
n_layers = 5

dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev, interface="torch", diff_method="best")
def quantum_circuit(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(n_qubits))
    qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
    return qml.probs(wires=range(n_qubits))

weight_shapes = {"weights": (n_layers, n_qubits, 3)}
qlayer = qml.qnn.TorchLayer(quantum_circuit, weight_shapes)

# Test default init
inputs = torch.randn(2, 8)
probs1 = qlayer(inputs)
ent1 = -(probs1 * torch.log(probs1 + 1e-9)).sum(dim=-1).mean()
print(f"Default Init Entropy: {ent1.item()}")

# Test near zero init
for p in qlayer.parameters():
    nn.init.normal_(p, mean=0.0, std=0.05)

probs2 = qlayer(inputs)
ent2 = -(probs2 * torch.log(probs2 + 1e-9)).sum(dim=-1).mean()
print(f"Near Zero Init Entropy: {ent2.item()}")

