"""
Quantum-Classical Hybrid: Self-Regulating Architect & Classical PINN Evaluator
==============================================================================
The Quantum Agent manages its own temperature and creativity internally.
The Classical Model has been completely converted into the Physics Evaluator (PINN).
We plot the Barren Plateau tracking.
"""

import torch
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import time

from agent_1_quantum_self_temp import QuantumArchitectSelfTemp
from turbine_pinn_geometry import CoolingChannelPINN, _get_final_design, plot_blade_design
from data_loader import get_cfd_dataloader

def calculate_grad_norm(model):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5


def run_hybrid_physics_agent(epochs=200, N=8):
    N_STATES = 2 ** N
    BATCH_SIZE = 8

    print("=" * 70)
    print("Self-Regulating Quantum Architect vs Classical Geometry PINN")
    print("Tracking Barren Plateaus and Self-Spiking Temperature")
    print("=" * 70)

    # Agent 1 (Quantum) creates probabilities
    q_agent = QuantumArchitectSelfTemp(n_qubits=N, n_layers=5)
    
    # Agent 2 (Classical) evaluates physical layout
    pinn = CoolingChannelPINN(input_dim=N_STATES, hidden_dim=128)

    params = list(q_agent.parameters()) + list(pinn.parameters())
    optimizer = optim.Adam(params, lr=0.003)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=2e-5)

    dataloader = get_cfd_dataloader(batch_size=BATCH_SIZE, samples=epochs * BATCH_SIZE, n_features=N)
    data_iter = iter(dataloader)

    losses = []
    grad_norms = []
    temperatures = []
    barren_flags = []

    start = time.time()

    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()

        try:
            env_ctx, cfd = next(data_iter)
            if env_ctx.shape[0] != BATCH_SIZE:
                p = BATCH_SIZE - env_ctx.shape[0]
                env_ctx = torch.cat([env_ctx, env_ctx[:p]], dim=0)
                cfd     = torch.cat([cfd,     cfd[:p]],     dim=0)
        except StopIteration:
            pass

        if cfd.shape[1] < 1:
            cfd = torch.zeros(cfd.shape[0], 1)

        # 1. Quantum architect proposes the design blueprint probabilities (Uses own temp)
        raw_probs, _ = q_agent(env_ctx)

        # 2. Classical PINN interprets probabilities into physical geometry features
        xc, yc, Rc, sh, dh = pinn(raw_probs)
        
        # 3. Classical PINN calculates multi-objective physical cost
        cost = pinn.calculate_cost_signal(xc, yc, Rc, sh, dh, cfd)

        # 4. Learning step
        losses.append(cost.item())
        cost.backward()
        
        # Calculate quantum gradient norm BEFORE clipping to log the 'true' signal strength
        g_norm = calculate_grad_norm(q_agent.qlayer)
        grad_norms.append(g_norm)
        
        torch.nn.utils.clip_grad_norm_(params, max_norm=5.0)
        optimizer.step()
        scheduler.step()

        # 5. Quantum Agent self-evaluates (checks for Barren Plateaus)
        q_agent.update_creativity(cost.item(), g_norm)
        temperatures.append(q_agent.temperature)
        barren_flags.append(q_agent.barren_plateau_flags[-1])
        
        if epoch % 50 == 0:
            print(f"Epoch {epoch:3d} | Cost: {cost.item():.2e} | "
                  f"GradNorm: {g_norm:.2e} | Temp: {q_agent.temperature:.2f} | "
                  f"Barren Plateau: {q_agent.barren_plateau_flags[-1]}")

    print(f"\nTraining completed in {(time.time()-start)/60:.1f} min")

    # =========================================================
    # Visualisation
    # =========================================================
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(2, 2)
    
    epochs_x = np.arange(1, epochs + 1)

    # Panel 1: Loss & Temperature
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(epochs_x, losses, color='#00d4ff', lw=2)
    ax1.set_yscale('log')
    ax1.set_ylabel('PINN Physical Cost (log)', color='#00d4ff')
    ax1.tick_params(axis='y', labelcolor='#00d4ff')
    ax1.set_xlabel('Epoch')
    ax1.set_title('Training Trajectory & Self-Regulation', fontsize=12)

    ax1_temp = ax1.twinx()
    ax1_temp.plot(epochs_x, temperatures, color='#ff9f43', lw=1.5, ls='--')
    ax1_temp.set_ylabel('Quantum Agent Temperature', color='#ff9f43')
    ax1_temp.tick_params(axis='y', labelcolor='#ff9f43')

    # Panel 2: Barren Plateau (Gradient Norm) Tracking
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(epochs_x, grad_norms, color='#ff4b5c', lw=2, label='Quantum Gradient Norm')
    
    # Highlight Barren Plateaus
    barren_blocks = np.where(barren_flags)[0]
    for b in barren_blocks:
        ax2.axvline(x=b+1, color='#ff9f43', alpha=0.3, lw=2)
        
    ax2.set_yscale('log')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Gradient Norm (log)')
    ax2.set_title('Barren Plateau Detection\n(Orange bars = Gradient Collapse Detected -> Temp Spike)', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.2)

    # Panel 3 & 4 (Combined): Final Geometrical Design
    # To use existing plotting code cleanly, we pass dummy context and extract params
    ax3 = fig.add_subplot(gs[1, :])
    q_xc, q_yc, q_Rc, q_sh, q_dh = _get_final_design(q_agent, pinn)
    plot_blade_design(ax3, q_xc, q_yc, q_Rc, q_sh, q_dh, pinn,
                      label='Self-Regulating Quantum Blueprint', color='#00d4ff')

    plt.tight_layout()
    plt.savefig('quantum_self_regulation_barren_plateau.png', dpi=250)
    print("Saved: quantum_self_regulation_barren_plateau.png")

if __name__ == "__main__":
    run_hybrid_physics_agent(epochs=250, N=8)
