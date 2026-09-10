"""
Turbine Blueprint Visualizer
============================
Extracts the final physical parameters chosen by the Quantum and Classical agents
to show exactly how they "designed" the turbine blade to beat the 7-objective physics.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from agent_1_quantum_architect import QuantumArchitect
from agent_1_classical_architect import ClassicalArchitect
from turbine_benchmark import TurbinePINN_v2
from turbine_benchmark_no_temp import run_turbine_agent_no_temp
from data_loader import get_cfd_dataloader

def extract_final_design():
    N = 10
    N_STATES = 2 ** N
    EPOCHS = 150
    BATCH_SIZE = 8

    print("=" * 70)
    print(f"Extracting Final Turbine Blueprint (N={N}, No Safety Net)")
    print("=" * 70)

    # 1. Train Quantum
    print("\nTraining Quantum Architect for design extraction...")
    qa = QuantumArchitect(n_qubits=N, n_layers=5)
    pinn_q = TurbinePINN_v2(input_dim=N_STATES, hidden_dim=128)
    run_turbine_agent_no_temp(qa, pinn_q, n_features=N, epochs=EPOCHS, batch_size=BATCH_SIZE)

    # 2. Train Classical
    print("Training Classical Architect for design extraction...")
    ca = ClassicalArchitect(n_features=N, n_states=N_STATES, hidden_dim=64)
    pinn_c = TurbinePINN_v2(input_dim=N_STATES, hidden_dim=128)
    run_turbine_agent_no_temp(ca, pinn_c, n_features=N, epochs=EPOCHS, batch_size=BATCH_SIZE)

    # 3. Generate Final Output
    dataloader = get_cfd_dataloader(batch_size=BATCH_SIZE, samples=BATCH_SIZE, n_features=N)
    env_ctx, cfd = next(iter(dataloader))
    if cfd.shape[1] < 3:
        cfd = torch.cat([cfd, torch.zeros(cfd.shape[0], 3 - cfd.shape[1])], dim=1)

    # Evaluate Quantum
    qa.eval()
    pinn_q.eval()
    with torch.no_grad():
        q_probs, _ = qa(env_ctx, temperature=1.0)
        q_params = pinn_q(q_probs)
        q_cost = pinn_q.calculate_cost_signal(*q_params, cfd).item()
        q_h, q_A, q_w, q_M, q_tbc, q_tip, q_geom = [p.mean().item() for p in q_params]

    # Evaluate Classical
    ca.eval()
    pinn_c.eval()
    with torch.no_grad():
        c_probs, _ = ca(env_ctx, temperature=1.0)
        c_params = pinn_c(c_probs)
        c_cost = pinn_c.calculate_cost_signal(*c_params, cfd).item()
        c_h, c_A, c_w, c_M, c_tbc, c_tip, c_geom = [p.mean().item() for p in c_params]

    # Normalize parameters for Radar Chart comparison
    # Approximate expected ranges based on PINN bounds
    bounds = {
        'Heat Transfer (h)': (200, 1000),      # W/m2K
        'Cooling Area (A)': (0.002, 0.05),     # m2
        'Blade Weight': (0.12, 0.6),           # kg
        'Blowing Ratio (M)': (0.3, 1.5),       # -
        'TBC Thickness': (0.05, 0.4),          # mm
        'Tip Clearance': (0.005, 0.03),        # -
        'Resonance Param': (0.0, 1.0)          # -
    }

    categories = list(bounds.keys())
    N_cat = len(categories)

    def normalize(val, bounds_tuple):
        v_min, v_max = bounds_tuple
        return max(0.0, min((val - v_min) / (v_max - v_min), 1.0))

    q_norm = [
        normalize(q_h, bounds['Heat Transfer (h)']),
        normalize(q_A, bounds['Cooling Area (A)']),
        normalize(q_w, bounds['Blade Weight']),
        normalize(q_M, bounds['Blowing Ratio (M)']),
        normalize(q_tbc, bounds['TBC Thickness']),
        normalize(q_tip, bounds['Tip Clearance']),
        normalize(q_geom, bounds['Resonance Param'])
    ]

    c_norm = [
        normalize(c_h, bounds['Heat Transfer (h)']),
        normalize(c_A, bounds['Cooling Area (A)']),
        normalize(c_w, bounds['Blade Weight']),
        normalize(c_M, bounds['Blowing Ratio (M)']),
        normalize(c_tbc, bounds['TBC Thickness']),
        normalize(c_tip, bounds['Tip Clearance']),
        normalize(c_geom, bounds['Resonance Param'])
    ]

    # --- Plotting the Radar Chart ---
    angles = [n / float(N_cat) * 2 * np.pi for n in range(N_cat)]
    angles += angles[:1]
    
    q_plot = q_norm + q_norm[:1]
    c_plot = c_norm + c_norm[:1]

    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    # X-axis ticks
    plt.xticks(angles[:-1], categories, color='white', size=11)
    
    # Y-axis ticks
    ax.set_rlabel_position(30)
    plt.yticks([0.25, 0.5, 0.75], ["Low", "Med", "High"], color="grey", size=10)
    plt.ylim(0, 1)

    # Quantum Plot
    ax.plot(angles, q_plot, linewidth=2, linestyle='solid', color='#00d4ff', label='Quantum Blueprint')
    ax.fill(angles, q_plot, '#00d4ff', alpha=0.25)

    # Classical Plot
    ax.plot(angles, c_plot, linewidth=2, linestyle='dashed', color='#ff4b5c', label='Classical Blueprint')
    ax.fill(angles, c_plot, '#ff4b5c', alpha=0.1)

    plt.title(f"Final Turbine Physical Design Strategy\n(N={N}, No Safety Net)", size=16, color='white', y=1.1)
    
    # Legend with actual physical values
    q_phys = f"Quantum Cost: {q_cost:.2e}\nh={q_h:.1f}, w={q_w:.2g}, M={q_M:.2f}, TBC={q_tbc:.2g}"
    c_phys = f"Classic Cost: {c_cost:.2e}\nh={c_h:.1f}, w={c_w:.2g}, M={c_M:.2f}, TBC={c_tbc:.2g}"
    
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), labels=[q_phys, c_phys])
    
    plt.tight_layout()
    plt.savefig('turbine_blueprint_radar.png', dpi=300)
    print("Saved: turbine_blueprint_radar.png")

if __name__ == "__main__":
    extract_final_design()
