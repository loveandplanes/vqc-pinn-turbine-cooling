import torch
import torch.nn as nn
import torch.nn.functional as F

class ClassicalSpecialistPINN(nn.Module):
    """
    Agent 2 (The Classical Specialist): A Physics-Informed Neural Network (PINN).
    It acts as the validator: digesting the design from Agent 1 and 'grading' it.
    This is an IMPARTIAL evaluator — it uses its own learned representation of
    thermodynamics to score both Quantum and Classical architects fairly.
    """
    def __init__(self, input_dim=64, hidden_dim=64):
        super(ClassicalSpecialistPINN, self).__init__()
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 3)  # Predicts: [h (conv. coeff), A (area), weight]
        )
        
    def forward(self, design_features):
        preds = self.net(design_features)
        h      = F.softplus(preds[:, 0])    # Heat transfer coefficient
        A      = F.softplus(preds[:, 1])    # Surface area of fractal channels
        weight = F.softplus(preds[:, 2])    # Structural weight
        return h, A, weight

    def calculate_cost_signal(self, h, A, weight, cfd_data, target_Q=2800.0, max_weight=1.2):
        """
        Core Physics-Informed Loss.
        Uses Q = h * A * ΔT from the CFD data.
        """
        delta_T = 390.0 - cfd_data[:, 0]           # Fuel cell temp - inlet temp
        Q_predicted = h * A * delta_T

        target = torch.tensor(target_Q)
        max_w  = torch.tensor(max_weight)

        # 1. Heat dissipation failure penalty
        thermal_loss = torch.relu(target - Q_predicted) ** 2

        # 2. Weight exceedance penalty
        weight_penalty = torch.exp(torch.relu(weight - max_w)) - 1.0

        # 3. Efficiency reward (maximize Q per unit mass)
        efficiency_reward = -0.1 * (Q_predicted / (weight + 1e-6))

        total_cost = (thermal_loss + weight_penalty + efficiency_reward).mean()
        return total_cost
