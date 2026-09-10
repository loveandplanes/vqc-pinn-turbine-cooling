import torch
import torch.optim as optim
import time

from agent_1_classical_architect import ClassicalArchitect
from agent_2_classical_specialist import ClassicalSpecialistPINN
from data_loader import get_cfd_dataloader
from fractal_visualizer import plot_3d_cooling_fractal

class ClassicalOptimizer:
    def __init__(self, n_features=6, hidden_dim=64):
        self.n_features = n_features
        self.n_states   = 2 ** n_features
        self.agent1_classical = ClassicalArchitect(n_features=n_features, n_states=self.n_states, hidden_dim=hidden_dim)
        self.agent2_classical = ClassicalSpecialistPINN(input_dim=self.n_states, hidden_dim=64)
        params = list(self.agent1_classical.parameters()) + list(self.agent2_classical.parameters())
        self.optimizer = optim.Adam(params, lr=0.015)

    def adjust_agent_creativity(self, loss_history, current_temp, current_top_p):
        if len(loss_history) < 5:
            return current_temp, current_top_p
        recent = loss_history[-5:]
        variance = sum(abs(recent[i] - recent[i-1]) for i in range(1, 5))
        if variance < 0.05 * abs(recent[-1]):
            return min(current_temp * 1.5, 3.0), min(current_top_p + 0.1, 1.0)
        return max(current_temp * 0.9, 0.1), max(current_top_p * 0.95, 0.5)

    def execute_handshake_protocol(self, epochs=50, batch_size=8, visualize=True):
        if visualize:
            print("Initializing Classical Pipeline...")
            time.sleep(0.5)
            print("Begin System Optimization!\n" + "="*50)

        dataloader    = get_cfd_dataloader(batch_size=batch_size, samples=epochs*batch_size,
                                           n_features=self.n_features)
        data_iterator = iter(dataloader)

        temperature = 1.0
        top_p = 0.8
        all_losses = []

        for epoch in range(1, epochs + 1):
            self.optimizer.zero_grad()
            try:
                env_context, raw_cfd_data = next(data_iterator)
                if env_context.shape[0] != batch_size:
                    p = batch_size - env_context.shape[0]
                    env_context   = torch.cat([env_context,   env_context[:p]],   dim=0)
                    raw_cfd_data  = torch.cat([raw_cfd_data,  raw_cfd_data[:p]],  dim=0)
            except StopIteration:
                pass

            raw_probs, _ = self.agent1_classical(env_context, temperature=temperature, top_p=top_p)
            h, A, weight = self.agent2_classical(raw_probs)
            cost_signal  = self.agent2_classical.calculate_cost_signal(h, A, weight, raw_cfd_data)
            all_losses.append(cost_signal.item())

            cost_signal.backward()
            self.optimizer.step()
            temperature, top_p = self.adjust_agent_creativity(all_losses, temperature, top_p)

            if visualize and (epoch % 5 == 0 or epoch == 1):
                print(f"Epoch {epoch:03d}/{epochs} | Cost: {cost_signal.item():.4f} | Temp: {temperature:.2f}")

        if visualize:
            print("="*50)
            print("Optimization Complete.")
            final_probs, _ = self.agent1_classical(env_context, temperature=0.05, top_p=0.2)
            plot_3d_cooling_fractal(final_probs[0], title="Optimal Classical Cooling Grid")

        return all_losses

if __name__ == "__main__":
    hub = ClassicalOptimizer(n_features=6, hidden_dim=64)
    hub.execute_handshake_protocol(epochs=50)
