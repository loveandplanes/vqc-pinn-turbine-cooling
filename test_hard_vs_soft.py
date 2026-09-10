"""
ROOT CAUSE DIAGNOSIS:
Gumbel-Softmax hard=True uses a straight-through estimator.
On the backward pass, gradient flows through the SOFT distribution.
When distribution is high-entropy (near-uniform), the softmax jacobian
d(soft_onehot)/d(logits) is ~(1/N) * I - essentially zero signal.
This means the quantum circuit CANNOT learn to shift probabilities.

FIX: Use hard=False during training (soft weighted selection from codebook),
and hard=True only at evaluation.
"""
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

# Fake codebook: reward state 5
codebook = torch.zeros(n_states)
codebook[5] = 1.0  # "good" design at index 5

optimizer = optim.Adam(qlayer.parameters(), lr=0.05)

print("=== hard=True (straight-through) ===")
for p in qlayer.parameters():
    nn.init.normal_(p, mean=0.0, std=0.02)
for epoch in range(100):
    optimizer.zero_grad()
    inputs = torch.randn(4, n_qubits)
    q_probs = qlayer(inputs)
    logits = torch.log(q_probs + 1e-9)
    one_hot = F.gumbel_softmax(logits, tau=0.5, hard=True)
    reward = (one_hot * codebook.unsqueeze(0)).sum(dim=-1).mean()
    loss = -reward
    loss.backward()
    optimizer.step()
if True:
    q_probs_eval = qlayer(torch.zeros(1, n_qubits))
    print(f"Final prob of state 5: {q_probs_eval[0,5].item():.4f}")
    print(f"Final entropy: {-(q_probs_eval * torch.log(q_probs_eval + 1e-9)).sum().item():.4f}")

print("\n=== hard=False (soft selection - smooth gradients) ===")
for p in qlayer.parameters():
    nn.init.normal_(p, mean=0.0, std=0.02)
optimizer = optim.Adam(qlayer.parameters(), lr=0.05)
for epoch in range(100):
    optimizer.zero_grad()
    inputs = torch.randn(4, n_qubits)
    q_probs = qlayer(inputs)
    logits = torch.log(q_probs + 1e-9)
    soft_weights = F.gumbel_softmax(logits, tau=0.5, hard=False)  # <-- KEY CHANGE
    reward = (soft_weights * codebook.unsqueeze(0)).sum(dim=-1).mean()
    loss = -reward
    loss.backward()
    optimizer.step()
if True:
    q_probs_eval = qlayer(torch.zeros(1, n_qubits))
    print(f"Final prob of state 5: {q_probs_eval[0,5].item():.4f}")
    print(f"Final entropy: {-(q_probs_eval * torch.log(q_probs_eval + 1e-9)).sum().item():.4f}")
