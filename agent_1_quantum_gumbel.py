"""
Quantum Architect with Gumbel-Softmax Discrete Design Selection
================================================================
Instead of passing a soft probability vector through an MLP (which erases
quantum advantage), this agent uses Gumbel-Softmax to make a DISCRETE
design choice from the quantum distribution.

The quantum circuit's probability distribution directly determines WHICH
design is selected. The gradient flows through the Gumbel-Softmax trick,
preserving end-to-end differentiability while maintaining the categorical
nature of the design choice.

This is the architecture that gives the quantum circuit a real job:
  - Classical MLP: unconstrained soft vector → MLP can learn anything
  - Quantum Gumbel: structured probability distribution → discrete choice
    The quantum correlations (entanglement) directly shape which designs
    are reachable, creating a fundamentally different inductive bias.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml


class QuantumArchitectGumbel(nn.Module):
    """
    Quantum Architect that uses Gumbel-Softmax for discrete design selection.
    Self-regulates temperature based on gradient flow (barren plateau detection).
    """
    def __init__(self, n_qubits: int = 4, n_layers: int = 3):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.n_states = 2 ** n_qubits

        # Quantum circuit
        self.dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(self.dev, interface="torch", diff_method="best")
        def quantum_circuit(inputs, weights):
            qml.AngleEmbedding(inputs, wires=range(self.n_qubits))
            qml.StronglyEntanglingLayers(weights, wires=range(self.n_qubits))
            return qml.probs(wires=range(self.n_qubits))

        weight_shapes = {"weights": (n_layers, n_qubits, 3)}
        self.qlayer = qml.qnn.TorchLayer(quantum_circuit, weight_shapes)
        
        # CRITICAL FIX 2: Initialize circuit near identity (zero rotation) to avoid Barren Plateaus.
        # Default initialization uses [0, 2pi], which starts the agent in a maximally chaotic 
        # (max entropy) state from which it cannot escape.
        for p in self.qlayer.parameters():
            nn.init.normal_(p, mean=0.0, std=0.05)

        # Trainable Design Codebook: each of the 2^n states maps to a
        # fixed set of continuous geometry parameters.
        # This is NOT an MLP — it's a lookup table trained end-to-end.
        # 27 = 5 channels * 3 params (xc, yc, Rc) + 6 holes * 2 params (sh, dh)
        self.codebook = nn.Parameter(torch.randn(self.n_states, 27) * 0.1)

        # FAIR-SHARPENING FIX: learnable logit scale.
        # Classical logits are unbounded (weight norm can grow -> arbitrarily
        # sharp delta). Quantum logits = log(q_probs) are bounded above by 0,
        # so without an explicit scale the quantum agent cannot sharpen its
        # distribution by growing a magnitude — it can only shrink T (which
        # kills gradients). This scalar restores parity: effective logits =
        # scale * log(q_probs), argmax-preserving, gap-amplifying.
        # Init 1.0 (= original behaviour); optimizer grows it to sharpen.
        self.logit_scale = nn.Parameter(torch.tensor(1.0))

        # Self-regulation state
        self.temperature = 1.0      # Gumbel temperature (high = soft, low = hard)
        self.gumbel_tau = 1.0       # Separate Gumbel-Softmax tau
        self.top_p = 0.8
        self.loss_history = []
        self.grad_history = []
        self.barren_plateau_flags = []
        self.entropy_history = []

    def update_creativity(self, loss_val: float, grad_norm: float):
        """
        Barren plateau detection using RELATIVE gradient collapse.
        No absolute threshold — adapts to whatever scale the loss is at.
        """
        self.loss_history.append(loss_val)
        self.grad_history.append(grad_norm)

        # Entropy of the current quantum distribution (logged for diagnostics)
        # (computed in forward, stored here)

        # BARREN PLATEAU DETECTION (RELATIVE):
        # If gradient drops to < 5% of its recent average
        is_barren = False
        if len(self.grad_history) > 10:
            avg_grad = sum(self.grad_history[-10:-1]) / 9
            if avg_grad > 0 and grad_norm < 0.05 * avg_grad:
                is_barren = True

        self.barren_plateau_flags.append(is_barren)

        if is_barren:
            # SHOCK: spike both temperatures to force exploration
            self.gumbel_tau = 5.0
            self.temperature = 5.0
            self.top_p = 1.0
        else:
            if len(self.loss_history) >= 5:
                recent = self.loss_history[-5:]
                variance = sum(abs(recent[i] - recent[i-1]) for i in range(1, 5))
                if variance < 0.05 * (abs(recent[-1]) + 1e-8) and recent[-1] > 0.02:
                    # Stuck in local minimum → warm up
                    self.gumbel_tau = min(self.gumbel_tau * 1.5, 3.0)
                    self.temperature = min(self.temperature * 1.5, 3.0)
                    self.top_p = min(self.top_p + 0.1, 1.0)
                else:
                    # Healthy descent or convergence → anneal toward hard categorical
                    self.gumbel_tau = max(self.gumbel_tau * 0.9, 0.05)
                    self.temperature = max(self.temperature * 0.9, 0.01)
                    self.top_p = max(self.top_p * 0.95, 0.8)

    def forward(self, env_context: torch.Tensor, **kwargs):
        """
        1. Quantum circuit produces probability distribution
        2. Temperature sharpening applied symmetrically with classical baseline
        3. Gumbel-Softmax samples a (nearly) one-hot design index
        4. Codebook lookup produces continuous geometry parameters
        """
        batch = env_context.shape[0]

        # Quantum probabilities from parameterized circuit
        q_probs = self.qlayer(env_context)           # (batch, n_states)

        # Map quantum probabilities to unbounded log-space logits
        # Apply learnable scale (clamped >= 1 so it can only sharpen, never
        # flatten below the original behaviour; cap avoids overflow).
        scale = torch.clamp(self.logit_scale, min=1.0, max=30.0)
        logits = torch.log(q_probs + 1e-9) * scale

        # Apply temperature sharpening symmetrically with classical agent
        probs = F.softmax(logits / (self.temperature + 1e-8), dim=-1)

        # Entropy for diagnostics
        entropy = -(probs * torch.log(probs + 1e-9)).sum(dim=-1).mean()
        self.entropy_history.append(entropy.item())

        # Gumbel-Softmax discrete sampling (uses the SAME scaled logits,
        # so training-time sampling and eval-time probs stay consistent)
        soft_onehot = F.gumbel_softmax(logits, tau=self.gumbel_tau, hard=True)
        # (batch, n_states)

        # Codebook lookup via soft attention
        # design_params = sum_i  soft_onehot_i * codebook_i
        design_params = soft_onehot @ self.codebook   # (batch, 27)

        return design_params, probs


class ClassicalArchitectGumbel(nn.Module):
    """
    Classical baseline with the SAME Gumbel-Softmax + codebook architecture.
    This ensures a fair comparison: the only difference is how the
    probability distribution is generated (MLP vs quantum circuit).
    """
    def __init__(self, n_features: int, n_states: int, hidden_dim: int = 16, matched_capacity: bool = True):
        super().__init__()
        self.n_states = n_states

        # Classical probability generator
        if matched_capacity:
            # Strictly matched capacity (~60-140 params) to align with quantum ansatz (36-90 params)
            h = min(hidden_dim, 8)
            self.prob_net = nn.Sequential(
                nn.Linear(n_features, h),
                nn.ReLU(),
                nn.Linear(h, n_states),
            )
        else:
            # Standard multi-layer overparameterized baseline
            self.prob_net = nn.Sequential(
                nn.Linear(n_features, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, n_states),
            )

        # SAME codebook as quantum agent
        self.codebook = nn.Parameter(torch.randn(n_states, 27) * 0.1)

        # Self-regulation state (same logic)
        self.temperature = 1.0
        self.gumbel_tau = 1.0
        self.top_p = 0.8
        self.loss_history = []
        self.grad_history = []
        self.barren_plateau_flags = []
        self.entropy_history = []

    def update_creativity(self, loss_val: float, grad_norm: float):
        self.loss_history.append(loss_val)
        self.grad_history.append(grad_norm)
        is_barren = False
        if len(self.grad_history) > 10:
            avg_grad = sum(self.grad_history[-10:-1]) / 9
            if avg_grad > 0 and grad_norm < 0.05 * avg_grad:
                is_barren = True
        self.barren_plateau_flags.append(is_barren)
        if is_barren:
            self.gumbel_tau = 5.0
            self.temperature = 5.0
        else:
            if len(self.loss_history) >= 5:
                recent = self.loss_history[-5:]
                variance = sum(abs(recent[i] - recent[i-1]) for i in range(1, 5))
                if variance < 0.05 * (abs(recent[-1]) + 1e-8) and recent[-1] > 0.02:
                    self.gumbel_tau = min(self.gumbel_tau * 1.5, 3.0)
                    self.temperature = min(self.temperature * 1.5, 3.0)
                else:
                    self.gumbel_tau = max(self.gumbel_tau * 0.9, 0.05)
                    self.temperature = max(self.temperature * 0.9, 0.01)

    def forward(self, env_context: torch.Tensor, **kwargs):
        batch = env_context.shape[0]
        logits = self.prob_net(env_context)                   # (batch, n_states)
        probs  = F.softmax(logits / (self.temperature + 1e-8), dim=-1)

        entropy = -(probs * torch.log(probs + 1e-9)).sum(dim=-1).mean()
        self.entropy_history.append(entropy.item())

        soft_onehot = F.gumbel_softmax(logits, tau=self.gumbel_tau, hard=True)
        design_params = soft_onehot @ self.codebook           # (batch, 27)
        return design_params, probs
