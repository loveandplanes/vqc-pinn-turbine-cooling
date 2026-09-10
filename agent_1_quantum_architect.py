import pennylane as qml
import torch
import torch.nn as nn

class QuantumArchitect(nn.Module):
    """
    Agent 1 (The Quantum Architect): A Variational Quantum Circuit (VQC) 
    designed to propose 3D-printed fractal-based cooling plate geometries.
    """
    def __init__(self, n_qubits=4, n_layers=3):
        super(QuantumArchitect, self).__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        
        # Simulated Quantum Device (can be swapped for actual hardware later)
        self.dev = qml.device("default.qubit", wires=self.n_qubits)
        
        # VQC Definition
        @qml.qnode(self.dev, interface="torch", diff_method="best")
        def quantum_circuit(inputs, weights):
            # Input Encoding: Map aerospace parameters (T, P) to qubits
            # Using AngleEmbedding to map values to rotations on the Bloch Sphere
            qml.AngleEmbedding(inputs, wires=range(self.n_qubits))
            
            # The Brain (Ansatz): Trainable parameters across superposition states
            qml.StronglyEntanglingLayers(weights, wires=range(self.n_qubits))
            
            # Return probabilities of each quantum state (the discrete design space)
            return qml.probs(wires=range(self.n_qubits))
        
        # Wrap QNode in a PyTorch Layer
        # StronglyEntanglingLayers expects weights of shape (layers, qubits, 3)
        weight_shapes = {"weights": (n_layers, n_qubits, 3)}
        self.vqc_layer = qml.qnn.TorchLayer(quantum_circuit, weight_shapes)

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
            physical_params: Tensor of shape (batch, n_qubits) containing T, P, etc.
            temperature, top_p: Creativity hyperparameters dynamically set by Agent 2
        """
        # 1. Get raw probabilities from the quantum circuit
        raw_probs = self.vqc_layer(physical_params)
        
        # 2. Tune the "Creativity" based on Agent 2's feedback
        sampled_probs = self.apply_creativity_filter(raw_probs, temperature, top_p)
        
        # 3. Sample an actual design index representing a parameter permutation 
        #    spanning the $2^n$ Hilbert space possible states.
        design_choices = torch.multinomial(sampled_probs, num_samples=1)
        
        return raw_probs, design_choices

if __name__ == "__main__":
    # Quick sanity check
    agent1 = QuantumArchitect(n_qubits=4, n_layers=3)
    # Physical inputs (e.g. normalized Temperature, Pressure, target thrust, etc.)
    dummy_inputs = torch.rand((2, 4))
    
    # Very "creative" mode
    raw, creative_design = agent1(dummy_inputs, temperature=2.0, top_p=0.9)
    print("Creative Design Samples:", creative_design.flatten().tolist())
    
    # Very "strict" mode
    raw, strict_design = agent1(dummy_inputs, temperature=0.1, top_p=0.5)
    print("Strict Design Samples:", strict_design.flatten().tolist())
