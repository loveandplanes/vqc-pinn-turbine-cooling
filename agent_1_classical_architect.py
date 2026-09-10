import torch
import torch.nn as nn
import torch.nn.functional as F

class ClassicalArchitect(nn.Module):
    """
    Agent 1 (The Classical Architect): A classical neural network (MLP)
    designed to propose 3D-printed fractal-based cooling plate geometries.
    This replaces the Variational Quantum Circuit (VQC) with a classical alternative.
    """
    def __init__(self, n_features=5, n_states=32, hidden_dim=64):
        super(ClassicalArchitect, self).__init__()
        self.n_features = n_features
        self.n_states = n_states
        
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_states)
        )

    def apply_creativity_filter(self, probs, temperature=1.0, top_p=1.0):
        """
        Adjusts the 'creativity' of the proposed designs.
        High temperature = more chaotic/exploration.
        Low temperature = more classical/exploitation.
        top_p = Restricts exploration to the most probable subset.
        """
        # Apply Temperature Scaling (Add a small epsilon to avoid log(0))
        logits = torch.log(probs + 1e-9) / temperature
        scaled_probs = torch.softmax(logits, dim=-1)
        
        # Nucleus Sampling (Top-p)
        sorted_probs, sorted_indices = torch.sort(scaled_probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        
        # Remove tokens (designs) with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        
        # Shift the indices to the right to keep the first token above threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        for i in range(scaled_probs.shape[0]):
            indices_to_remove = sorted_indices[i][sorted_indices_to_remove[i]]
            scaled_probs[i, indices_to_remove] = 0.0
            
        # Re-normalize to get valid probability distribution
        final_probs = scaled_probs / torch.sum(scaled_probs, dim=-1, keepdim=True)
        return final_probs

    def forward(self, physical_params, temperature=1.0, top_p=1.0):
        """
        Args:
            physical_params: Tensor of shape (batch, n_features) containing T, P, etc.
            temperature, top_p: Creativity hyperparameters dynamically set by Agent 2
        """
        # 1. Get raw probabilities from the classical neural network
        logits = self.net(physical_params)
        raw_probs = torch.softmax(logits, dim=-1)
        
        # 2. Tune the "Creativity" based on Agent 2's feedback
        sampled_probs = self.apply_creativity_filter(raw_probs, temperature, top_p)
        
        # 3. Sample an actual design index representing a parameter permutation 
        #    spanning the $2^n$ Hilbert space possible states.
        design_choices = torch.multinomial(sampled_probs, num_samples=1)
        
        return raw_probs, design_choices

if __name__ == "__main__":
    # Quick sanity check
    agent1 = ClassicalArchitect(n_features=5, n_states=32)
    dummy_inputs = torch.rand((2, 5))
    
    # Very "creative" mode
    raw, creative_design = agent1(dummy_inputs, temperature=2.0, top_p=0.9)
    print("Creative Design Samples:", creative_design.flatten().tolist())
    
    # Very "strict" mode
    raw, strict_design = agent1(dummy_inputs, temperature=0.1, top_p=0.5)
    print("Strict Design Samples:", strict_design.flatten().tolist())
