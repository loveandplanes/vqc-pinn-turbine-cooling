"""
Rigorous Quantum vs Classical Benchmark — Fixed Codebook Search
================================================================
Fixes 3 fundamental flaws in the previous benchmark:

1. REDUCED QUBITS: 4 qubits → 16 states (quantum can shape distribution)
2. FROZEN CODEBOOK: Pre-computed diverse designs, NOT trainable.
   Both agents must SEARCH, not optimize a single entry.
3. MATCHED PARAMETERS: Classical MLP has ~same param count as quantum circuit.

The key insight: when the codebook is trainable, the classical agent
collapses to entropy=0 (one design) and just optimizes that entry.
This makes the problem a trivial 27-dim continuous optimization where
quantum has NO advantage. By freezing the codebook, we force a true
combinatorial search over discrete designs — quantum's native domain.

Context-Dependent Search:
  Different environmental conditions (gas temp, coolant flow, altitude)
  require different optimal designs. The agent must learn a MAPPING from
  context → best design index. Quantum entanglement creates correlations
  that let the circuit represent complex context→design mappings with
  far fewer parameters than classical networks.
"""

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import time

from agent_1_quantum_gumbel import QuantumArchitectGumbel, ClassicalArchitectGumbel
from turbine_pinn_geometry import CoolingChannelPINN, N_CH, N_H
from data_loader import get_cfd_dataloader


def calculate_grad_norm(model):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5


def precompute_frozen_codebook(n_states, pinn):
    """
    Generate a FIXED codebook with 1 Global Minimum (Golden Design) 
    and 15 Sub-optimal/Deceptive Local Minima.
    """
    import math
    torch.manual_seed(42)
    codebook = torch.zeros(n_states, 27)

    for i in range(n_states):
        phase = i / n_states
        # Create deceptive/bad designs
        noise = torch.randn(27) * 0.5
        for c in range(N_CH):
            base_x = (c + phase) / (N_CH + 1)
            codebook[i, c*3 + 0] = base_x * 2.0 - 1.0 + noise[c*3]
            codebook[i, c*3 + 1] = 0.8 + noise[c*3 + 1]  # Bad yc (often outside blade)
            codebook[i, c*3 + 2] = noise[c*3 + 2]
        for h in range(N_H):
            base_s = (h + phase) / (N_H + 1)
            codebook[i, N_CH*3 + h*2]     = base_s * 2.0 - 1.0 + noise[N_CH*3 + h*2]
            codebook[i, N_CH*3 + h*2 + 1] = noise[N_CH*3 + h*2 + 1]

    # THE "GOLDEN" DESIGNS (Indices 8, 9, 10, 11) - Context-Dependent Minima
    # Perfect Mass Balance for different target COM x-positions (0.35, 0.40, 0.45, 0.50)
    # Indices chosen to be well within n_states (works for both N=5/32 and N=8/256)
    golden_indices = [8, 9, 10, 11]
    com_x_targets = [0.35, 0.40, 0.45, 0.50]
    
    for idx, target_com_x in zip(golden_indices, com_x_targets):
        shift = target_com_x - 0.45
        target_xcs = [0.15 + shift, 0.30 + shift, 0.45 + shift, 0.60 + shift, 0.75 + shift]
        target_ycs = [0.01, 0.01, 0.01, 0.01, 0.01]   # mean = 0.01 (Perfect COM and safe)
        
        for c in range(N_CH):
            xc, yc = target_xcs[c], target_ycs[c]
            # Ensure xc is within physical bounds [0.06, 0.80] to avoid math domain errors
            xc = max(0.061, min(0.799, xc))
            raw_x = -math.log(0.74 / (xc - 0.06) - 1.0)
            val = (yc - 0.02) / 0.055
            raw_y = 0.5 * math.log((1.0 + val) / (1.0 - val))
            val2 = (0.015 - 0.006) / 0.010
            raw_R = math.log(math.exp(val2) - 1.0)
            
            codebook[idx, c*3 + 0] = raw_x
            codebook[idx, c*3 + 1] = raw_y
            codebook[idx, c*3 + 2] = raw_R

        # Calibrated Film Holes for Golden Design:
        # Distributed across LE, suction, and pressure sides with dh ≈ 0.0062 (M ≈ 1.12 optimal peak)
        golden_sh = [0.05, 0.15, 0.28, 0.45, 0.65, 0.85]
        for h in range(N_H):
            sh_target = golden_sh[h]
            raw_sh = -math.log(1.0 / sh_target - 1.0)
            val_dh = (0.0062 - 0.001) / 0.003
            raw_dh = math.log(math.exp(val_dh) - 1.0)
            codebook[idx, N_CH*3 + h*2] = raw_sh
            codebook[idx, N_CH*3 + h*2 + 1] = raw_dh

    return codebook


def precompute_codebook_costs(frozen_codebook, pinn, n_cfd_samples=16, transform="log_row"):
    """
    Precompute the physics cost of every codebook entry.
    Uses a grid of representative CFD conditions covering the operational range.
    Returns cost_table: Tensor of shape (n_cfd_samples, N_STATES).

    transform="log_row" (copy fix, default): log(cost) then per-pressure-row
    min-max. The old global min-max on raw costs divided everything by the
    catastrophic-design range (~2.1M), compressing the 751%-in-raw-units gap
    between 'collapse on idx 10' and 'track context' into 0.016 normalized
    units — unlearnable. Log + per-row scaling restores it to ~0.31 (19x).
    transform="global" reproduces the original scaling.
    """
    N_STATES = frozen_codebook.shape[0]
    # Sample a grid of CFD pressure values across the operational range [0.5, 1.2]
    pressures = torch.linspace(0.5, 1.2, n_cfd_samples)
    cost_table = torch.zeros(n_cfd_samples, N_STATES)

    print(f"  Precomputing cost table ({n_cfd_samples} x {N_STATES})...", end="", flush=True)
    with torch.no_grad():
        for p_idx, pressure in enumerate(pressures):
            # Representative CFD row: [Inlet_Temp=340K, Pressure, Flow=10, HeatFlux=2.25]
            cfd_row = torch.tensor([[340.0, pressure.item(), 10.0, 2.25]])
            for i in range(N_STATES):
                dp = frozen_codebook[i:i+1]
                ch = dp[:, :N_CH*3].reshape(-1, N_CH, 3)
                h  = dp[:, N_CH*3:].reshape(-1, N_H,  2)
                xc = torch.sigmoid(ch[:, :, 0]) * 0.74 + 0.06
                yc = torch.tanh   (ch[:, :, 1]) * 0.055 + 0.02
                Rc = F.softplus   (ch[:, :, 2]) * 0.010 + 0.006
                sh = torch.sigmoid(h[:, :, 0])
                dh = F.softplus   (h[:, :, 1]) * 0.003 + 0.001
                cost_table[p_idx, i] = pinn.calculate_cost_signal(
                    xc, yc, Rc, sh, dh, cfd_row).item()
    print(" done.")
    if transform == "log_row":
        # Compress the 7e3 dynamic range, then scale each pressure row so its
        # best design = 0 and worst = 1 (argmin per row unchanged: monotonic).
        log_table = torch.log(cost_table + 1.0)
        rmin = log_table.min(dim=1, keepdim=True).values
        rmax = log_table.max(dim=1, keepdim=True).values
        cost_table = (log_table - rmin) / (rmax - rmin + 1e-8)
    else:
        # Original global [0, 1] normalization (kept for reproducibility).
        c_min = cost_table.min()
        c_max = cost_table.max()
        cost_table = (cost_table - c_min) / (c_max - c_min + 1e-8)
    return cost_table, pressures


def run_frozen_codebook_agent(agent, pinn, frozen_codebook, n_features, epochs,
                               is_dynamic=True, cost_table=None, cost_pressures=None,
                               final_anneal_epochs=50, anneal_temp_floor=0.05,
                               anneal_tau_floor=0.1):
    """
    Training loop with FROZEN codebook.
    The agent learns to SELECT the best design from a fixed menu.

    KEY FIX: We train on EXPECTED COST = probs @ cost_table[context]
    rather than the cost of a single Gumbel-sampled design.
    This gives a dense, smooth gradient to the circuit on every step.

    FINAL FORCED ANNEAL (copy fix): during the last `final_anneal_epochs`
    epochs, temperature and Gumbel tau are cosine-decayed to their floors
    and self-regulation shocks are disabled. This forces the distribution
    to sharpen into a deployable delta instead of staying hot forever
    (the failure mode behind Quantum E[cost]=0.0669 vs Classical 0.0230:
    the variance trigger kept re-heating T whenever progress stalled).
    Applies equally to quantum and classical agents (fair).
    """
    import math
    N_STATES = frozen_codebook.shape[0]
    BATCH_SIZE = 8
    # Clamp so short verification runs still explore before annealing.
    eff_anneal = max(0, min(final_anneal_epochs, epochs // 2))
    anneal_start = epochs - eff_anneal + 1  # first epoch of forced anneal
    anneal_temp_start = None
    anneal_tau_start = None

    # Freeze the codebook
    with torch.no_grad():
        agent.codebook.copy_(frozen_codebook)
    agent.codebook.requires_grad_(False)

    # Only optimize agent's distribution parameters — NOT PINN, NOT codebook.
    # If PINN params are optimized, it adapts the cost landscape to accept bad designs.
    trainable_params = [p for p in agent.parameters() if p.requires_grad]
    optimizer = optim.Adam(trainable_params, lr=0.02)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-4)

    dataloader = get_cfd_dataloader(batch_size=BATCH_SIZE,
                                    samples=epochs * BATCH_SIZE,
                                    n_features=n_features)
    data_iter = iter(dataloader)

    history = {
        'loss': [], 'grad_norm': [], 'temp': [], 'gumbel_tau': [],
        'barren': [], 'entropy': [], 'selected_designs': [],
        'discrete_cost': [],
    }

    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()

        try:
            env_ctx, cfd = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            env_ctx, cfd = next(data_iter)

        if env_ctx.shape[0] != BATCH_SIZE:
            p = BATCH_SIZE - env_ctx.shape[0]
            env_ctx = torch.cat([env_ctx, env_ctx[:p]], dim=0)
            cfd     = torch.cat([cfd,     cfd[:p]], dim=0)

        # Get probability distribution from agent (no Gumbel sampling for training)
        _, probs = agent(env_ctx)   # probs: (batch, N_STATES)

        # Track selected designs (argmax for logging)
        selected_idx = probs.argmax(dim=-1)
        selected = selected_idx.tolist()
        history['selected_designs'].extend(selected)

        # --- Expected Physical Cost = probs @ cost_vector(context) ---
        if cost_table is not None and cost_pressures is not None:
            pressure_vals = cfd[:, 1].clamp(0.5, 1.2)   # (batch,)
            # Find nearest precomputed pressure for each sample
            p_idx = torch.argmin(
                (cost_pressures.unsqueeze(0) - pressure_vals.unsqueeze(1)).abs(), dim=1
            )  # (batch,)
            costs_per_sample = cost_table[p_idx]   # (batch, N_STATES)
            expected_cost = (probs * costs_per_sample).sum(dim=-1).mean()
            discrete_cost = costs_per_sample[torch.arange(BATCH_SIZE), selected_idx].mean()
            history['discrete_cost'].append(discrete_cost.item())
        else:
            # Fallback: sample one design via Gumbel (much noisier)
            logits = torch.log(probs + 1e-9)
            one_hot = F.gumbel_softmax(logits, tau=agent.gumbel_tau, hard=True)
            dp = one_hot @ frozen_codebook
            ch = dp[:, :N_CH*3].reshape(-1, N_CH, 3)
            h  = dp[:, N_CH*3:].reshape(-1, N_H, 2)
            xc = torch.sigmoid(ch[:, :, 0]) * 0.74 + 0.06
            yc = torch.tanh   (ch[:, :, 1]) * 0.055 + 0.02
            Rc = F.softplus   (ch[:, :, 2]) * 0.010 + 0.006
            sh = torch.sigmoid(h[:, :, 0])
            dh = F.softplus   (h[:, :, 1]) * 0.003 + 0.001
            expected_cost = pinn.calculate_cost_signal(xc, yc, Rc, sh, dh, cfd)
            history['discrete_cost'].append(expected_cost.item())

        entropy = -(probs * torch.log(probs + 1e-9)).sum(dim=-1).mean()
        total_loss = expected_cost
        history['loss'].append(expected_cost.item())
        total_loss.backward()

        g_norm = calculate_grad_norm(agent)
        history['grad_norm'].append(g_norm)

        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=5.0)
        optimizer.step()
        scheduler.step()

        if is_dynamic and eff_anneal > 0 and epoch >= anneal_start:
            # FINAL FORCED ANNEAL: cosine-decay T/tau to floors, no shocks.
            if anneal_temp_start is None:
                anneal_temp_start = agent.temperature
                anneal_tau_start = agent.gumbel_tau
            progress = (epoch - anneal_start + 1) / eff_anneal  # 0..1
            cosine = 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
            agent.temperature = anneal_temp_floor + (anneal_temp_start - anneal_temp_floor) * cosine
            agent.gumbel_tau = anneal_tau_floor + (anneal_tau_start - anneal_tau_floor) * cosine
            agent.barren_plateau_flags.append(False)
        elif is_dynamic:
            agent.update_creativity(expected_cost.item(), g_norm)

        history['temp'].append(agent.temperature)
        history['gumbel_tau'].append(agent.gumbel_tau)
        history['barren'].append(
            agent.barren_plateau_flags[-1] if len(agent.barren_plateau_flags) > 0 else False
        )
        history['entropy'].append(
            agent.entropy_history[-1] if len(agent.entropy_history) > 0 else 0.0
        )

        if epoch % 50 == 0:
            n_unique = len(set(history['selected_designs'][-50*BATCH_SIZE:]))
            print(f"  Epoch {epoch:4d} | E[Cost]: {expected_cost.item():.4f} | "
                  f"Deployed: {history['discrete_cost'][-1]:.4f} | "
                  f"GradNorm: {g_norm:.2e} | Temp: {agent.temperature:.2f} | "
                  f"Tau: {agent.gumbel_tau:.2f} | Entropy: {history['entropy'][-1]:.2f} | "
                  f"Unique designs (last 50): {n_unique}")

    return history


if __name__ == "__main__":
    EPOCHS = 500
    N = 5          # VALIDATION: 5 qubits → 32 states. Confirms architecture is correct before scaling.
    N_STATES = 2 ** N
    C_HIDDEN = 8   # Classical hidden dim

    # Parameter count check
    dummy_q = QuantumArchitectGumbel(n_qubits=N, n_layers=4)
    dummy_c = ClassicalArchitectGumbel(n_features=N, n_states=N_STATES, hidden_dim=C_HIDDEN, matched_capacity=True)
    q_circuit_params = sum(p.numel() for p in dummy_q.qlayer.parameters())
    c_mlp_params = sum(p.numel() for p in dummy_c.prob_net.parameters())
    print(f"Parameter count (distribution generator only):")
    print(f"  Quantum circuit: {q_circuit_params} params")
    print(f"  Classical MLP:   {c_mlp_params} params")
    print(f"  Codebook:        {N_STATES * 27} params (FROZEN, shared)")

    print("=" * 70)
    print("RIGOROUS Benchmark: Frozen Codebook Combinatorial Search")
    print(f"Quantum circuit -> Gumbel-Softmax -> FROZEN Codebook[{N_STATES}] -> Physics")
    print(f"N={N} ({N_STATES} states) | {EPOCHS} epochs | Classical hidden={C_HIDDEN}")
    print("=" * 70)

    # Pre-compute frozen codebook (shared by all agents)
    dummy_pinn = CoolingChannelPINN(input_dim=N_STATES, hidden_dim=128)
    frozen_cb = precompute_frozen_codebook(N_STATES, dummy_pinn)
    print(f"\nFrozen codebook shape: {frozen_cb.shape}")

    # Pre-compute cost table for all designs across all CFD conditions
    # This is the key ingredient that gives dense gradients to both agents.
    print("\nPrecomputing physics cost table (one-time cost)...")
    cost_table, cost_pressures = precompute_codebook_costs(frozen_cb, dummy_pinn, n_cfd_samples=16)
    print(f"  Cost table shape: {cost_table.shape}")
    print(f"  Best design per pressure: {cost_table.argmin(dim=1).tolist()}")

    total_start = time.time()

    # --- 1. Quantum Agent (Self-Regulating) ---
    print("\n[1/4] Quantum Gumbel (Self-Regulating)...")
    torch.manual_seed(42)
    q_agent = QuantumArchitectGumbel(n_qubits=N, n_layers=4)
    q_pinn  = CoolingChannelPINN(input_dim=N_STATES, hidden_dim=128)
    hist_q_dynamic = run_frozen_codebook_agent(
        q_agent, q_pinn, frozen_cb.clone(), N, EPOCHS, is_dynamic=True,
        cost_table=cost_table, cost_pressures=cost_pressures)
    print(f"  Final E[cost]: {hist_q_dynamic['loss'][-1]:.4f}")

    # --- 2. Quantum Agent (Fixed Temperature) ---
    print("\n[2/4] Quantum Gumbel (Fixed T=0.01)...")
    torch.manual_seed(42)
    q_fixed = QuantumArchitectGumbel(n_qubits=N, n_layers=4)
    q_fixed.temperature = 0.01
    q_fixed.gumbel_tau = 0.1
    q_pinn_f = CoolingChannelPINN(input_dim=N_STATES, hidden_dim=128)
    hist_q_fixed = run_frozen_codebook_agent(
        q_fixed, q_pinn_f, frozen_cb.clone(), N, EPOCHS, is_dynamic=False,
        cost_table=cost_table, cost_pressures=cost_pressures)
    print(f"  Final E[cost]: {hist_q_fixed['loss'][-1]:.4f}")

    # --- 3. Classical Agent (Self-Regulating) ---
    print("\n[3/4] Classical Gumbel (Self-Regulating)...")
    torch.manual_seed(42)
    c_agent = ClassicalArchitectGumbel(n_features=N, n_states=N_STATES, hidden_dim=C_HIDDEN, matched_capacity=True)
    c_pinn  = CoolingChannelPINN(input_dim=N_STATES, hidden_dim=128)
    hist_c_dynamic = run_frozen_codebook_agent(
        c_agent, c_pinn, frozen_cb.clone(), N, EPOCHS, is_dynamic=True,
        cost_table=cost_table, cost_pressures=cost_pressures)
    print(f"  Final E[cost]: {hist_c_dynamic['loss'][-1]:.4f}")

    # --- 4. Classical Agent (Fixed Temperature) ---
    print("\n[4/4] Classical Gumbel (Fixed T=0.01)...")
    torch.manual_seed(42)
    c_fixed = ClassicalArchitectGumbel(n_features=N, n_states=N_STATES, hidden_dim=C_HIDDEN, matched_capacity=True)
    c_fixed.temperature = 0.01
    c_fixed.gumbel_tau = 0.1
    c_pinn_f = CoolingChannelPINN(input_dim=N_STATES, hidden_dim=128)
    hist_c_fixed = run_frozen_codebook_agent(
        c_fixed, c_pinn_f, frozen_cb.clone(), N, EPOCHS, is_dynamic=False,
        cost_table=cost_table, cost_pressures=cost_pressures)
    print(f"  Final E[cost]: {hist_c_fixed['loss'][-1]:.4f}")

    elapsed = (time.time() - total_start) / 60
    print(f"\nAll 4 agents completed in {elapsed:.1f} min")

    # =========================================================================
    # FAIR EVAL (fix in copy): deployed argmax cost at fixed low T for all.
    # E[cost] = probs @ costs rewards collapse; the manufactured blade is the
    # argmax design, so report cost(argmax) per context at T=0.01 for both.
    # =========================================================================
    def fair_deployed_eval(agent, n_batches=50, batch_size=8):
        agent.temperature = 0.01
        agent.gumbel_tau = 0.1
        dl = get_cfd_dataloader(batch_size=batch_size,
                                samples=n_batches * batch_size, n_features=N)
        deployed, entropies, picks = [], [], []
        agent.eval()
        with torch.no_grad():
            for env_ctx, cfd in dl:
                _, probs = agent(env_ctx)
                idx = probs.argmax(dim=-1)
                pvals = cfd[:, 1].clamp(0.5, 1.2)
                p_idx = torch.argmin(
                    (cost_pressures.unsqueeze(0) - pvals.unsqueeze(1)).abs(), dim=1)
                costs = cost_table[p_idx]
                deployed.append(costs[torch.arange(len(idx)), idx].mean().item())
                entropies.append((-(probs * torch.log(probs + 1e-9)).sum(dim=-1)).mean().item())
                picks.extend(idx.tolist())
        return float(np.mean(deployed)), float(np.mean(entropies)), len(set(picks))

    print("\n" + "=" * 70)
    print("FAIR EVAL @ T=0.01 — deployed (argmax) cost, lower = better:")
    fair_results = {}
    for name, ag in [("Q Self-Reg", q_agent), ("Q Fixed", q_fixed),
                     ("C Self-Reg", c_agent), ("C Fixed", c_fixed)]:
        dep, ent, div = fair_deployed_eval(ag)
        fair_results[name] = (dep, ent, div)
        print(f"  {name:12s}: deployed={dep:.4f} entropy={ent:.2f} unique={div}")
    print("=" * 70)
    with open('fair_eval.txt', 'w') as f:
        for name, (dep, ent, div) in fair_results.items():
            f.write(f"{name}: deployed={dep:.6f} entropy={ent:.4f} unique={div}\n")
    print("Saved: fair_eval.txt")

    # =========================================================================
    # Visualization
    # =========================================================================
    plt.style.use('dark_background')
    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    fig.suptitle(
        'Rigorous Benchmark: Frozen Codebook Combinatorial Search\n'
        f'N={N} ({N_STATES} states) | Quantum ({q_circuit_params}p) vs '
        f'Classical ({c_mlp_params}p) | Codebook FROZEN',
        fontsize=13, y=0.98
    )

    epochs_x = np.arange(1, EPOCHS + 1)
    from scipy.ndimage import uniform_filter1d
    smooth = lambda x, w=15: uniform_filter1d(np.array(x, dtype=float), size=w)

    # --- Panel 1: Loss Comparison ---
    ax = axes[0, 0]
    ax.semilogy(epochs_x, smooth(hist_q_dynamic['loss']), color='#00d4ff', lw=2.5,
                label='Quantum (Self-Reg)')
    ax.semilogy(epochs_x, smooth(hist_q_fixed['loss']), color='#00d4ff', lw=1.5,
                ls='--', alpha=0.7, label='Quantum (Fixed)')
    ax.semilogy(epochs_x, smooth(hist_c_dynamic['loss']), color='#2ecc71', lw=2.5,
                label='Classical (Self-Reg)')
    ax.semilogy(epochs_x, smooth(hist_c_fixed['loss']), color='#2ecc71', lw=1.5,
                ls='--', alpha=0.7, label='Classical (Fixed)')
    ax.set_title('Convergence (Smoothed)', fontsize=11)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Physical Cost (log)')
    ax.legend(fontsize=9); ax.grid(True, ls=':', alpha=0.3)

    # --- Panel 2: Entropy ---
    ax = axes[0, 1]
    ax.plot(epochs_x, smooth(hist_q_dynamic['entropy']), color='#00d4ff', lw=2,
            label='Quantum (Self-Reg)')
    ax.plot(epochs_x, smooth(hist_q_fixed['entropy']), color='#00d4ff', lw=1.5,
            ls='--', alpha=0.7, label='Quantum (Fixed)')
    ax.plot(epochs_x, smooth(hist_c_dynamic['entropy']), color='#2ecc71', lw=2,
            label='Classical (Self-Reg)')
    ax.plot(epochs_x, smooth(hist_c_fixed['entropy']), color='#2ecc71', lw=1.5,
            ls='--', alpha=0.7, label='Classical (Fixed)')
    ax.axhline(y=np.log(N_STATES), color='white', ls=':', alpha=0.3, label=f'Max entropy (ln {N_STATES})')
    ax.set_title('Distribution Entropy\n(↓=collapsed, ↑=exploring)', fontsize=11)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Entropy (nats)')
    ax.legend(fontsize=8); ax.grid(True, ls=':', alpha=0.3)

    # --- Panel 3: Gradient Norm ---
    ax = axes[0, 2]
    ax.semilogy(epochs_x, smooth(hist_q_dynamic['grad_norm']), color='#00d4ff', lw=2,
                label='Quantum (Self-Reg)')
    ax.semilogy(epochs_x, smooth(hist_q_fixed['grad_norm']), color='#00d4ff', lw=1.5,
                ls='--', alpha=0.7, label='Quantum (Fixed)')
    ax.semilogy(epochs_x, smooth(hist_c_dynamic['grad_norm']), color='#2ecc71', lw=2,
                label='Classical (Self-Reg)')
    ax.semilogy(epochs_x, smooth(hist_c_fixed['grad_norm']), color='#2ecc71', lw=1.5,
                ls='--', alpha=0.7, label='Classical (Fixed)')
    for b in np.where(hist_q_dynamic['barren'])[0]:
        ax.axvline(x=b+1, color='#ff9f43', alpha=0.2, lw=2)
    ax.set_title('Gradient Norm\n(Orange = Barren Plateau Detected)', fontsize=11)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Gradient Norm (log)')
    ax.legend(fontsize=8); ax.grid(True, ls=':', alpha=0.3)

    # --- Panel 4: Temperature Self-Regulation ---
    ax = axes[1, 0]
    ax.plot(epochs_x, hist_q_dynamic['temp'], color='#00d4ff', lw=2,
            label='Quantum Temp')
    ax.plot(epochs_x, hist_q_dynamic['gumbel_tau'], color='#ff9f43', lw=1.5,
            ls='--', label='Quantum Gumbel τ')
    ax.plot(epochs_x, hist_c_dynamic['temp'], color='#2ecc71', lw=1.5,
            label='Classical Temp')
    ax.set_title('Self-Regulation: Temperature & Gumbel τ', fontsize=11)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Value')
    ax.set_ylim(0, 6); ax.legend(fontsize=9); ax.grid(True, ls=':', alpha=0.3)

    # --- Panel 5: Design Diversity (unique designs explored) ---
    ax = axes[1, 1]
    window = 40 * 8  # 40 epochs × batch_size
    def rolling_unique(designs, w):
        result = []
        for i in range(0, len(designs), 8):
            start = max(0, i - w)
            result.append(len(set(designs[start:i+8])))
        return result

    q_div = rolling_unique(hist_q_dynamic['selected_designs'], window)
    c_div = rolling_unique(hist_c_dynamic['selected_designs'], window)
    div_x = np.linspace(1, EPOCHS, len(q_div))
    ax.plot(div_x, q_div, color='#00d4ff', lw=2, label='Quantum (Self-Reg)')
    ax.plot(div_x, c_div, color='#2ecc71', lw=2, label='Classical (Self-Reg)')
    ax.axhline(y=N_STATES, color='white', ls=':', alpha=0.3, label=f'Max ({N_STATES} designs)')
    ax.set_title('Design Diversity\n(Unique designs explored per window)', fontsize=11)
    ax.set_xlabel('Epoch'); ax.set_ylabel('# Unique Designs')
    ax.legend(fontsize=9); ax.grid(True, ls=':', alpha=0.3)

    # --- Panel 6: Final Cost Bar Chart ---
    ax = axes[1, 2]
    final_costs = {
        'Q Self-Reg': hist_q_dynamic['loss'][-1],
        'Q Fixed': hist_q_fixed['loss'][-1],
        'C Self-Reg': hist_c_dynamic['loss'][-1],
        'C Fixed': hist_c_fixed['loss'][-1],
    }
    colors_bar = ['#00d4ff', '#0090ff', '#2ecc71', '#1f9955']
    bars = ax.bar(final_costs.keys(), final_costs.values(), color=colors_bar,
                  edgecolor='white', linewidth=0.5)
    ax.set_title('Final Physical Cost\n(Lower = Better)', fontsize=11)
    ax.set_ylabel('Cost')
    ax.set_xticks(range(4))
    ax.set_xticklabels(final_costs.keys(), rotation=20, ha='right', fontsize=9)
    for bar, val in zip(bars, final_costs.values()):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{val:.2e}', ha='center', va='bottom', fontsize=9, color='white')

    plt.tight_layout()
    plt.savefig('rigorous_benchmark.png', dpi=250, bbox_inches='tight')
    print("Saved: rigorous_benchmark.png")

    # =========================================================================
    # Visualisation: Blade Geometries
    # =========================================================================
    from turbine_pinn_geometry import plot_blade_design
    from collections import Counter

    fig2, axes2 = plt.subplots(1, 2, figsize=(16, 6))
    fig2.suptitle('Refined Multi-Physics Blade Geometries: Quantum vs. Classical', fontsize=16)

    # Decode function for a specific codebook index
    def decode_index(idx):
        design = frozen_cb[idx].unsqueeze(0)
        ch = design[:, :N_CH*3].reshape(-1, N_CH, 3)
        h  = design[:, N_CH*3:].reshape(-1, N_H,  2)
        xc = torch.sigmoid(ch[:, :, 0]) * 0.74 + 0.06
        yc = torch.tanh   (ch[:, :, 1]) * 0.055 + 0.02
        Rc = F.softplus   (ch[:, :, 2]) * 0.010 + 0.006
        sh = torch.sigmoid(h[:, :, 0])
        dh = F.softplus   (h[:, :, 1]) * 0.003 + 0.001
        return xc[0].numpy(), yc[0].numpy(), Rc[0].numpy(), sh[0].numpy(), dh[0].numpy()

    # Representative CFD condition (P=0.85 atm)
    cfd_eval = torch.tensor([[340.0, 0.85, 10.0, 2.25]])

    # Quantum Agent Best Design
    q_best_idx = Counter(hist_q_dynamic['selected_designs'][-400:]).most_common(1)[0][0]
    q_xc, q_yc, q_Rc, q_sh, q_dh = decode_index(q_best_idx)
    with torch.no_grad():
        _, q_diag = dummy_pinn.calculate_cost_signal(
            torch.tensor(q_xc).unsqueeze(0), torch.tensor(q_yc).unsqueeze(0),
            torch.tensor(q_Rc).unsqueeze(0), torch.tensor(q_sh).unsqueeze(0),
            torch.tensor(q_dh).unsqueeze(0), cfd_eval, return_diagnostics=True)

    plot_blade_design(axes2[0], q_xc, q_yc, q_Rc, q_sh, q_dh, dummy_pinn, 
                      label=f'Quantum Best (Idx {q_best_idx})', color='#00d4ff')
    axes2[0].set_title(
        f"Quantum Designer (Index {q_best_idx})\n"
        f"T_max: {q_diag['T_metal_max_K']:.1f} K | Pump: {q_diag['pumping_power_W']:.0f} W | Stress Ratio: {q_diag['stress_ratio']:.2f}",
        fontsize=11
    )

    # Classical Agent Best Design
    c_best_idx = Counter(hist_c_dynamic['selected_designs'][-400:]).most_common(1)[0][0]
    c_xc, c_yc, c_Rc, c_sh, c_dh = decode_index(c_best_idx)
    with torch.no_grad():
        _, c_diag = dummy_pinn.calculate_cost_signal(
            torch.tensor(c_xc).unsqueeze(0), torch.tensor(c_yc).unsqueeze(0),
            torch.tensor(c_Rc).unsqueeze(0), torch.tensor(c_sh).unsqueeze(0),
            torch.tensor(c_dh).unsqueeze(0), cfd_eval, return_diagnostics=True)

    plot_blade_design(axes2[1], c_xc, c_yc, c_Rc, c_sh, c_dh, dummy_pinn, 
                      label=f'Classical Best (Idx {c_best_idx})', color='#2ecc71')
    axes2[1].set_title(
        f"Classical Designer (Index {c_best_idx})\n"
        f"T_max: {c_diag['T_metal_max_K']:.1f} K | Pump: {c_diag['pumping_power_W']:.0f} W | Stress Ratio: {c_diag['stress_ratio']:.2f}",
        fontsize=11
    )

    plt.tight_layout()
    plt.savefig('rigorous_geometries.png', dpi=250, bbox_inches='tight')
    print("Saved: rigorous_geometries.png")
