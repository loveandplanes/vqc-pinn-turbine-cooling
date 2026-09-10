import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml

class QuantumArchitectSelfTemp(nn.Module):
    """
    Quantum Architect with Self-Regulating Temperature & Creativity.
    It manages its own `temperature` and `top_p` based on gradient flow.
    If it detects a 'Barren Plateau' (vanishing gradients), it spikes its creativity.
    """
    def __init__(self, n_qubits: int = 4, n_layers: int = 3):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.n_states = 2 ** n_qubits

        self.dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(self.dev, interface="torch", diff_method="best")
        def quantum_circuit(inputs, weights):
            qml.AngleEmbedding(inputs, wires=range(self.n_qubits))
            qml.StronglyEntanglingLayers(weights, wires=range(self.n_qubits))
            return qml.probs(wires=range(self.n_qubits))

        weight_shapes = {"weights": (n_layers, n_qubits, 3)}
        self.qlayer = qml.qnn.TorchLayer(quantum_circuit, weight_shapes)

        # Self-managed state
        self.temperature = 1.0
        self.top_p = 0.8
        self.loss_history = []
        self.grad_history = []  # Added to track signal strength
        self.barren_plateau_flags = []

    def _apply_creativity(self, logits: torch.Tensor):
        probs = F.softmax(logits / self.temperature, dim=-1)
        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        core_mask = cumulative_probs < self.top_p
        core_mask[..., 1:] = core_mask[..., :-1].clone()
        core_mask[..., 0] = True
        
        filtered_probs = sorted_probs * core_mask.float()
        filtered_probs = filtered_probs / filtered_probs.sum(dim=-1, keepdim=True)
        
        final_probs = torch.zeros_like(probs)
        final_probs.scatter_(-1, sorted_indices, filtered_probs)
        return final_probs

    def update_creativity(self, loss_val: float, grad_norm: float):
        """
        Detects Barren Plateaus (Relative) and adjusts internal temperature.
        """
        self.loss_history.append(loss_val)
        self.grad_history.append(grad_norm)
        
        # BARREN PLATEAU DETECTION (RELATIVE): 
        # If signal drops to < 5% of its recent average, it's a collapse.
        is_barren = False
        if len(self.grad_history) > 10:
            avg_grad = sum(self.grad_history[-10:-1]) / 9
            if grad_norm < 0.05 * avg_grad and loss_val > 1e7:
                is_barren = True
            
        self.barren_plateau_flags.append(is_barren)

        if is_barren:
            # SHOCK THE SYSTEM: Massive creativity spike
            self.temperature = 5.0
            self.top_p = 1.0
        else:
            if len(self.loss_history) >= 5:
                recent = self.loss_history[-5:]
                variance = sum(abs(recent[i] - recent[i-1]) for i in range(1, 5))
                # Normal local minimum detection
                if variance < 0.05 * (abs(recent[-1]) + 1e-8):
                    self.temperature = min(self.temperature * 1.5, 3.0)
                    self.top_p = min(self.top_p + 0.1, 1.0)
                else:
                    # Healthy descent -> Cool down to exploit
                    self.temperature = max(self.temperature * 0.9, 0.01)
                    self.top_p = max(self.top_p * 0.95, 0.8)

    def forward(self, env_context: torch.Tensor, **kwargs):
        q_probs = self.qlayer(env_context)  # (batch, N_states)
        logits  = torch.log(q_probs + 1e-9)
        
        final_probs = self._apply_creativity(logits)
        return final_probs, q_probs
