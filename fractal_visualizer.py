import numpy as np
import matplotlib.pyplot as plt

def plot_3d_cooling_fractal(quantum_states_prob, title="Optimized 3D Fractal Cooling Plate"):
    """
    Maps the localized quantum superposition into a 3D branching fractal pattern.
    High probability states dictate thicker branches or denser cooling channels.
    """
    import torch
    if isinstance(quantum_states_prob, torch.Tensor):
        probs = quantum_states_prob.detach().cpu().numpy().flatten()
    else:
        probs = np.array(quantum_states_prob).flatten()

    n_states = len(probs)
    # Ensure the grid covers at least n_states by finding appropriate dimensions
    cols = 8 if n_states >= 8 else 4
    rows = int(np.ceil(n_states / cols))

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Aesthetic dark mode setup for holographic visual context
    ax.set_facecolor('black')
    fig.patch.set_facecolor('black')

    # Phi and Theta generation for primary directional vectors radially outward 
    # We use linspace and tile/repeat to ensure we have exactly n_states points
    phi_vals = np.linspace(0, np.pi, rows)
    theta_vals = np.linspace(0, 2.0*np.pi, cols)
    
    phi, theta = np.meshgrid(phi_vals, theta_vals)
    phi = phi.flatten()[:n_states]
    theta = theta.flatten()[:n_states]
    
    # Generate the branching logic based on Agent 1's Quantum Probability Profile
    for i in range(n_states):
        # Probability dictates physical prominence of this branch constraint
        prominence = probs[i] * 50 
        
        # Endpoints of the Primary Cooling Branch
        x1 = prominence * np.sin(phi[i]) * np.cos(theta[i])
        y1 = prominence * np.sin(phi[i]) * np.sin(theta[i])
        z1 = prominence * np.cos(phi[i])
        
        # Color intensity and line thickness is driven by Quantum state likelihood
        color_intensity = min(1.0, probs[i] * 5.0)
        color = plt.cm.inferno(color_intensity) # Using inferno colormap for thermal aesthetics
        linewidth = max(1.0, probs[i] * 15.0)
        
        # Plot Primary
        ax.plot([0, x1], [0, y1], [0, z1], color=color, linewidth=linewidth, alpha=0.9)
        
        # Secondary branching (Fractal nature to mimic vascular pathways)
        for j in range(2):
            dx = x1 + (prominence * 0.5) * np.sin(phi[i] + j*0.5) * np.cos(theta[i] + j*0.5)
            dy = y1 + (prominence * 0.5) * np.sin(phi[i] + j*0.5) * np.sin(theta[i] + j*0.5)
            dz = z1 + (prominence * 0.5) * np.cos(phi[i] + j*0.5)
            
            ax.plot([x1, dx], [y1, dy], [z1, dz], color=color, linewidth=linewidth*0.6, alpha=0.7)
            
            # Tertiary branching (micro-channels)
            for k in range(2):
                ddx = dx + (prominence * 0.25) * np.sin(phi[i] + k*0.3) * np.cos(theta[i] - k*0.3)
                ddy = dy + (prominence * 0.25) * np.sin(phi[i] + k*0.3) * np.sin(theta[i] - k*0.3)
                ddz = dz + (prominence * 0.25) * np.cos(phi[i] + k*0.3)
                
                ax.plot([dx, ddx], [dy, ddy], [dz, ddz], color=color, linewidth=linewidth*0.3, alpha=0.5)

    ax.set_title(title, color='white', fontsize=16)
    ax.set_xlabel('X (Micro-Axis)', color='gray')
    ax.set_ylabel('Y (Micro-Axis)', color='gray')
    ax.set_zlabel('Z (Micro-Axis)', color='gray')
    ax.tick_params(colors='gray')
    
    plt.tight_layout()
    # Save the output visualization directly to the directory
    plt.savefig("fractal_design_output.png", dpi=300, facecolor='black')
    print("Agent completed design. Holographic blueprint saved to 'fractal_design_output.png'")
