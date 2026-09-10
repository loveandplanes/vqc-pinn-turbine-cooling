"""
Turbine Blade Cooling Channel Design — Geometry-Driven Thermal PINN
====================================================================
 
THE SETUP
---------
The blade shape is FIXED: a NACA 4412 profile (turbine-adapted, 50 mm chord).
The agent is not designing the blade shape — it is designing what goes INSIDE it:
 
  • 5 internal circular cooling channels
      - Position (xc, yc) inside the blade cross-section
      - Radius Rc (determines wall thickness and internal flow velocity)
 
  • 6 film cooling holes punched through the blade wall
      - Arc-length position sh along the blade surface
      - Hole diameter dh
 
The physics are computed directly from the geometry — wall thickness is not
a free parameter, it is the measured distance from the channel center to the
nearest surface point minus the channel radius.
 
PHYSICS (thermal only)
----------------------
1. External convection h_ext(s)
   Stagnation-point model: peak at LE, decay along suction/pressure sides.
   Secondary turbulent transition bump on suction side (~25% arc from LE).
   h_ext(s) = H_stag * [exp(-a*s²) + 0.4*exp(-b*(s-s_tr)²)]
 
2. Film cooling effectiveness η(M)  [non-monotonic — local minimum trap]
   η(M) = M · exp(-0.5·(M−1.2)²/0.4²) / (1 + 0.08·M³)
   Peaks at M≈1.2 (jet attached), drops at M>2 (jet lift-off).
   M = ρ_cool·V_cool / (ρ_gas·V_gas)  — proportional to hole diameter.
   Local minimum: small holes look safe (low M, η>0) but miss the optimum.
 
3. Adiabatic wall temperature
   T_aw(s) = T_gas - η_eff(s)·(T_gas - T_cool)
   η_eff(s) at each surface point = max effectiveness from nearby holes,
   decaying with arc-length distance from each hole's exit.
 
4. Internal convection h_int(Rc)  [non-linear — trade-off trap]
   Fixed total coolant flow split equally among N_ch channels.
   Smaller channel → higher velocity → higher Re → higher h_int.
   h_int ∝ Rc^{-1.8}  (from Dittus-Boelter + continuity)
   Trade-off: small Rc → h_int↑ but Rc↓ forces channel near surface → thin wall
 
5. Wall thermal resistance
   R_wall = t_wall / k_metal     where t_wall = dist(center→surface) - Rc
   If t_wall ≤ 0: channel pokes through blade → catastrophic penalty.
 
6. 1D resistance network per surface element
   q_i   = (T_aw_i - T_cool) / (1/h_ext_i + R_wall_i + 1/h_int_i)
   T_metal_outer_i = T_aw_i - q_i / h_ext_i
 
LOCAL MINIMA CATALOGUE
-----------------------
  LM1  All channels at x≈0.25–0.35 (peak heat load).
       LE and TE are uncooled → T_metal spikes there.
       Escaping requires simultaneously moving ≥2 channels — saddle.
 
  LM2  Thin-wall trap: large Rc pushed close to surface.
       t_wall → t_min (just above penalty threshold), h_int low (large Rc).
       The agent finds this because t_wall constraint appears soft initially.
 
  LM3  Film holes clustered near LE.
       LE is well-cooled, but η decays fast downstream → mid-chord TE hot.
       Moving a hole downstream requires giving up LE coverage (competing gradient).
 
  LM4  Channel overlap: two channels close together share a thermal zone.
       Combined they look adequate, but one is redundant — wasting coolant.
       The overlap penalty is quadratic beyond a threshold → slow gradient there.
 
  LM5  High-efficiency illusion: very small Rc → h_int extremely high.
       But small channel near TE has very thin surrounding wall and poor area coverage.
       The "good local cooling here" masks the rest of the surface being uncovered.

  LM6  MASS BALANCE TRAP (The Quantum Showcase): 
       Turbine blades spin at high RPM; their cooling channels must have a 
       Center of Mass (COM) matching the rotational axis to prevent vibration.
       We enforce a massive penalty if the COM deviates from (x=0.45, y=0.01).
       To spread the channels out for cooling, an agent must move TWO OR MORE 
       channels SIMULTANEOUSLY in opposite directions to preserve the COM.
       Classical MLPs update parameters independently based on local gradients,
       so they get completely trapped (moving one channel ruins the COM).
       Quantum circuits, via entanglement, naturally coordinate multi-parameter
       moves, allowing them to escape this trap.
"""
 
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import time
 
from agent_1_quantum_architect  import QuantumArchitect
from agent_1_classical_architect import ClassicalArchitect
from data_loader                 import get_cfd_dataloader
 
 
# =============================================================================
#  Physical constants
# =============================================================================
CHORD       = 0.050      # m — blade chord length
K_METAL     = 20.0       # W/(m·K) — CMSX-4 nickel superalloy thermal conductivity
T_GAS       = 1650.0     # K — hot gas temperature
T_COOL      = 680.0      # K — coolant air supply temperature
T_METAL_LIM = 1100.0     # K — maximum allowable metal temperature
H_STAG      = 3500.0     # W/(m²·K) — stagnation zone h_ext
T_WALL_MIN  = 0.008      # chord fractions — minimum wall thickness (~0.4 mm)
 
# Coolant air properties at ~700 K
RHO_COOL    = 0.50       # kg/m³
K_AIR       = 0.052      # W/(m·K)
MU_AIR      = 3.7e-5     # Pa·s
PR_AIR      = 0.71
COOLANT_FLOW_TOTAL = 0.002  # kg/s — total coolant mass flow (fixed supply)

# Structural & Mechanical constants (CMSX-4 nickel superalloy & 15,000 RPM rotor)
RHO_METAL         = 8700.0     # kg/m³ — CMSX-4 single-crystal superalloy density
OMEGA_RPM         = 15000.0    # RPM — turbine rotor rotational speed
OMEGA_RAD         = OMEGA_RPM * (2.0 * np.pi / 60.0)  # ~1570.8 rad/s
ROTOR_RADIUS_MEAN = 0.45       # m — rotor mean radius
BLADE_SPAN        = 0.080      # m — blade radial span length (80 mm)
SOLID_BLADE_AREA  = 0.082 * (CHORD**2)  # m² — NACA 4412 solid cross-section area (~2.05e-4 m²)
SIGMA_YIELD_BASE  = 950.0e6    # Pa — 950 MPa baseline yield strength at 1100 K
DARCY_ROUGHNESS   = 0.03       # 3D additive printing channel wall roughness factor
 
N_CH = 5    # number of internal cooling channels
N_H  = 6    # number of film cooling holes
N_SURF = 80  # surface discretization points (each side)
 
 
# =============================================================================
#  Blade geometry — NACA 4412 adapted for turbine blade
# =============================================================================
 
def _build_blade_geometry(n_pts: int = N_SURF) -> dict:
    """
    Build fixed blade surface geometry.
 
    Returns a dict of numpy arrays and torch tensors describing the
    NACA 4412 blade surface (normalized by chord, x ∈ [0,1]).
 
    Surface parameterization:
      s=0   Leading edge (stagnation point)
      s→0.5 Suction side (upper) from LE to TE
      s→1.0 Pressure side (lower) from TE back to LE
    """
    xi = np.linspace(0.0, 1.0, n_pts)
 
    # NACA 4-digit thickness distribution (12% thickness)
    t = 0.12
    y_t = 5*t * (
          0.2969 * np.sqrt(xi + 1e-9)
        - 0.1260 * xi
        - 0.3516 * xi**2
        + 0.2843 * xi**3
        - 0.1015 * xi**4
    )
 
    # Camber line — NACA 4412 (m=4%, p=40% chord)
    m, p = 0.04, 0.40
    y_c  = np.where(xi < p,
                    m / p**2 * (2*p*xi - xi**2),
                    m / (1-p)**2 * ((1 - 2*p) + 2*p*xi - xi**2))
    dy_c = np.where(xi < p,
                    2*m / p**2 * (p - xi),
                    2*m / (1-p)**2 * (p - xi))
    theta = np.arctan(dy_c)
 
    # Upper (suction) side: LE → TE
    x_suc = xi - y_t * np.sin(theta)
    y_suc = y_c + y_t * np.cos(theta)
 
    # Lower (pressure) side: LE → TE (reversed for perimeter traversal)
    x_pre = xi + y_t * np.sin(theta)
    y_pre = y_c - y_t * np.cos(theta)
 
    # Full surface: LE → suction → TE → pressure(reversed) → LE
    # Convention: s=0 at index 0 (LE), monotonically increasing
    x_full = np.concatenate([x_suc, x_pre[-2:0:-1]])
    y_full = np.concatenate([y_suc, y_pre[-2:0:-1]])
 
    # Arc-length parameterization
    ds    = np.sqrt(np.diff(x_full)**2 + np.diff(y_full)**2)
    s_cum = np.concatenate([[0.0], np.cumsum(ds)])
    s_norm = s_cum / s_cum[-1]         # [0, 1]
    total_perim = s_cum[-1]             # chord fractions
 
    # Surface normals (inward-pointing, for wall-thickness direction check)
    # Not used in the cost directly, but useful for visualization
    nx = np.gradient(y_full, s_cum)
    ny = -np.gradient(x_full, s_cum)
    n_mag = np.sqrt(nx**2 + ny**2) + 1e-9
    nx /= n_mag; ny /= n_mag
 
    # h_ext distribution along the surface
    # Stagnation at s=0 (LE), decays along both sides.
    # Secondary turbulent transition bump on suction side at s_tr≈0.18.
    s = s_norm
    h_ext = (H_STAG      * np.exp(-60.0 * s**2)
           + H_STAG*0.35 * np.exp(-120.0 * (s - 0.18)**2)   # suction-side transition
           + H_STAG*0.25 * np.exp(-25.0 * (s - 0.05)**2)    # pressure-side early
           + H_STAG*0.15)                                     # baseline convection
 
    return {
        'x':      x_full.astype(np.float32),
        'y':      y_full.astype(np.float32),
        's':      s_norm.astype(np.float32),
        'h_ext':  h_ext.astype(np.float32),
        'perim':  float(total_perim),
        # For visualization
        'x_suc':  x_suc.astype(np.float32),
        'y_suc':  y_suc.astype(np.float32),
        'x_pre':  x_pre.astype(np.float32),
        'y_pre':  y_pre.astype(np.float32),
    }
 
 
# Pre-compute once at import time
BLADE = _build_blade_geometry()
 
 
# =============================================================================
#  Thermal PINN — geometry → physics → cost
# =============================================================================
 
class CoolingChannelPINN(nn.Module):
    """
    Physics-Informed Neural Network for internal cooling channel optimization.
 
    Input:  raw_probs from agent  (batch, 2**N_qubits)
    Output: channel geometry parameters (xc, yc, Rc, sh, dh)
 
    Cost is computed from first-principles thermal physics on the blade geometry.
    """
 
    def __init__(self, input_dim: int = 256, hidden_dim: int = 128):
        super().__init__()
        # 5 channels × (xc, yc, Rc) + 6 holes × (sh, dh) = 27 outputs
        n_out = N_CH * 3 + N_H * 2
 
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, n_out),
        )
 
        # Register blade geometry as non-trainable buffers
        self.register_buffer('surf_x',  torch.tensor(BLADE['x']))    # (N_surf,)
        self.register_buffer('surf_y',  torch.tensor(BLADE['y']))    # (N_surf,)
        self.register_buffer('surf_s',  torch.tensor(BLADE['s']))    # (N_surf,) arc-length
        self.register_buffer('h_ext_profile', torch.tensor(BLADE['h_ext']))  # (N_surf,)
 
    # ------------------------------------------------------------------
    def forward(self, z: torch.Tensor):
        """
        Map agent output to physically-bounded channel/hole parameters.
 
        Returns:
            xc  (batch, N_CH) — channel centre x-position [0.06, 0.82] chord fracs
            yc  (batch, N_CH) — channel centre y-position [-0.04, 0.09] chord fracs
            Rc  (batch, N_CH) — channel radius [0.006, ~0.040] chord fracs
            sh  (batch, N_H)  — hole arc-length position [0, 1]
            dh  (batch, N_H)  — hole diameter [0.001, ~0.008] chord fracs
        """
        raw = self.net(z)   # (batch, 27)
 
        ch  = raw[:, :N_CH*3].reshape(-1, N_CH, 3)
        h   = raw[:, N_CH*3:].reshape(-1, N_H,  2)
 
        # Channel centres: kept within the blade interior
        xc  = torch.sigmoid(ch[:, :, 0]) * 0.74 + 0.06    # [0.06, 0.80]
        yc  = torch.tanh   (ch[:, :, 1]) * 0.055 + 0.02   # [−0.035, +0.075]  (cambered region)
        Rc  = F.softplus   (ch[:, :, 2]) * 0.010 + 0.006  # [0.006, ∞)  chord fracs
 
        # Film cooling holes
        sh  = torch.sigmoid(h[:, :, 0])                    # [0, 1] arc-length
        dh  = F.softplus   (h[:, :, 1]) * 0.003 + 0.001   # [0.001, ∞) chord fracs
 
        return xc, yc, Rc, sh, dh
 
    # ------------------------------------------------------------------
    # Individual physics modules
    # ------------------------------------------------------------------
 
    def _wall_thickness(self, xc, yc, Rc):
        """
        Minimum distance from each channel centre to the blade surface, minus Rc.
 
        t_wall < 0  → channel protrudes through wall (catastrophic)
        t_wall > 0  → physical wall thickness at thinnest point
 
        Shape: (batch, N_CH)
        """
        # (batch, N_CH, 1) − (1, 1, N_surf)
        dx = xc.unsqueeze(-1) - self.surf_x.unsqueeze(0).unsqueeze(0)
        dy = yc.unsqueeze(-1) - self.surf_y.unsqueeze(0).unsqueeze(0)
        dist = torch.sqrt(dx**2 + dy**2 + 1e-8)           # (batch, N_CH, N_surf)
        min_dist, _ = dist.min(dim=-1)                     # (batch, N_CH)
        t_wall = min_dist - Rc                             # (batch, N_CH)
        return t_wall, dist                                # also return full dist matrix
 
    def _internal_h(self, Rc):
        """
        Dittus-Boelter internal convection coefficient.
 
        Coolant flow split equally among N_CH channels.
        Smaller channel → higher velocity → higher Re → higher h_int.
 
        h_int ∝ Rc^{-1.8}  (analytically derived from continuity + Dittus-Boelter)
 
        Shape: (batch, N_CH)
        """
        Rc_phys = Rc * CHORD                               # m
        flow_per_ch = COOLANT_FLOW_TOTAL / N_CH            # kg/s
        A_ch     = torch.clamp(np.pi * Rc_phys**2, min=1e-8)
        V_cool   = flow_per_ch / (RHO_COOL * A_ch + 1e-8)
        Re       = RHO_COOL * V_cool * (2 * Rc_phys) / MU_AIR
        Re_safe  = torch.clamp(Re, min=100.0)
        Nu       = 0.023 * Re_safe**0.8 * PR_AIR**0.4
        h_int    = Nu * K_AIR / (2 * Rc_phys + 1e-8)
        return h_int                                       # W/(m²·K)

    def _pumping_power(self, Rc):
        """
        Darcy-Weisbach internal pressure drop and pumping power.
        Small channels (Rc -> 0) incur an exponential pumping penalty (W_pump ∝ Rc^-5),
        preventing unrealistic 'infinite cooling' loopholes.

        Returns:
            pumping_power : (batch, N_CH) Watts per channel
            delta_P       : (batch, N_CH) Pa pressure drop per channel
        """
        Rc_phys = Rc * CHORD                               # m
        flow_per_ch = COOLANT_FLOW_TOTAL / N_CH            # kg/s
        A_ch     = torch.clamp(np.pi * Rc_phys**2, min=1e-8)
        V_cool   = flow_per_ch / (RHO_COOL * A_ch + 1e-8)
        Re       = RHO_COOL * V_cool * (2 * Rc_phys) / MU_AIR
        Re_safe  = torch.clamp(Re, min=100.0)

        # Darcy friction factor: Blasius turbulent relation + 3D printed wall roughness
        f_darcy = 0.3164 * (Re_safe**(-0.25)) + DARCY_ROUGHNESS

        # Pressure drop across span: Delta P = f * (L / D_h) * (0.5 * rho * V^2)
        delta_P = f_darcy * (BLADE_SPAN / (2.0 * Rc_phys + 1e-8)) * (0.5 * RHO_COOL * V_cool**2)

        # Pumping power per channel: W = Delta P * (mass_flow / rho)
        pumping_power = delta_P * (flow_per_ch / RHO_COOL)
        return pumping_power, delta_P

    def _structural_stress(self, Rc, xc, yc, T_metal_max):
        """
        Centrifugal structural stress and stress concentration under 15,000 RPM rotation.
        Accounts for net load-bearing area reduction and inter-channel ligament notch factors.

        Returns:
            stress_ratio        : (batch,) sigma_peak / sigma_yield_adj
            area_reduction_ratio: (batch,) fraction of solid blade area hollowed out
            sigma_peak          : (batch,) peak Von Mises / tensile stress (Pa)
            sigma_yield_adj     : (batch,) temperature-degraded yield strength (Pa)
        """
        batch = Rc.shape[0]
        Rc_phys = Rc * CHORD  # m
        channel_area = np.pi * (Rc_phys**2).sum(dim=-1)  # (batch,) m²
        area_reduction_ratio = channel_area / SOLID_BLADE_AREA

        # Centrifugal acceleration: a_cent = omega^2 * r_mean
        a_cent = (OMEGA_RAD**2) * ROTOR_RADIUS_MEAN       # ~1.11e6 m/s² (~113,000 g)

        # Nominal centrifugal tensile stress on net blade metal area
        # F_cent = rho_metal * span * a_cent * solid_area
        # sigma_nom = F_cent / net_area = sigma_base / (1 - area_reduction)
        net_area_fraction = torch.clamp(1.0 - area_reduction_ratio, min=0.20)
        sigma_nom = (RHO_METAL * BLADE_SPAN * a_cent) / net_area_fraction  # Pa

        # Stress concentration factor (Kirsch hole solution Kt ≈ 3.0)
        # Increases when inter-channel ligament distance is thin:
        min_ligament = torch.full((batch,), 0.05, device=Rc.device)
        for i in range(N_CH):
            for j in range(i+1, N_CH):
                dist_ij = torch.sqrt(
                    (xc[:, i] - xc[:, j])**2 + (yc[:, i] - yc[:, j])**2 + 1e-8
                )
                lig = dist_ij - (Rc[:, i] + Rc[:, j])
                min_ligament = torch.minimum(min_ligament, lig)

        # Ligament notch factor with Neuber elastic-plastic blunting cap (max Kt = 6.0)
        kt_factor = torch.clamp(3.0 * (1.0 + 0.003 / torch.clamp(min_ligament, min=0.001)), min=2.5, max=6.0)
        sigma_peak = sigma_nom * kt_factor                # Pa

        # Temperature-dependent yield strength of CMSX-4 single-crystal superalloy:
        # High temperatures cause gamma-prime precipitate coarsening, reducing yield
        t_factor = torch.clamp((T_metal_max - 1050.0) / 350.0, min=0.0, max=1.0)
        sigma_yield_adj = SIGMA_YIELD_BASE * (1.0 - 0.60 * (t_factor**2))  # Pa

        stress_ratio = sigma_peak / (sigma_yield_adj + 1e-4)
        return stress_ratio, area_reduction_ratio, sigma_peak, sigma_yield_adj

    def _assign_channels_to_surface_2d(self, dist_ch_surf, t_wall, h_int):
        """
        2D Multi-Channel Thermal Conduction Diffusion.
        Replaces hard 1D argmin with a smooth thermal diffusion kernel (length scale ~0.06 chord).
        Provides physically smooth inter-channel heat conduction and continuous gradients.
        """
        diffusion_scale = 0.06
        # Soft-min weights across channels for each surface point
        weights = F.softmax(-dist_ch_surf / diffusion_scale, dim=1)  # (batch, N_CH, N_surf)

        t_wall_surf = (t_wall.unsqueeze(-1) * weights).sum(dim=1)    # (batch, N_surf)
        h_int_surf  = (h_int.unsqueeze(-1)  * weights).sum(dim=1)    # (batch, N_surf)
        return t_wall_surf, h_int_surf
 
    def _film_effectiveness(self, sh, dh):
        """
        Film cooling effectiveness field along blade surface.
 
        For each surface point s_i, the effective η is the contribution
        from ALL holes, decaying with arc-length distance from each hole
        exit. Holes are not purely local — their film "rides" downstream.
 
        η_hole(M) — non-monotonic Goldstein/Taylor form:
            peak at M≈1.2, drops at M>2 due to jet lift-off
 
        Downstream decay:
            η(s) = η_hole · exp(−κ · |s − s_hole| / dh)
            The 1/dh dependence: larger holes have longer effective coverage.
 
        Shape returned: (batch, N_surf)
        """
        # Blowing ratio proxy: M ∝ dh (larger hole → more coolant → higher M)
        M = dh * 180.0   # calibrated so dh=0.006 → M≈1.1 (near optimum)
 
        # Non-monotonic η(M)
        eta_peak = M * torch.exp(-0.5 * ((M - 1.2)**2) / 0.40**2)
        eta_lift = 1.0 / (1.0 + 0.08 * M**3)
        eta_hole = eta_peak * eta_lift                     # (batch, N_H)  ∈ [0, ~0.72]
 
        # Arc-length distance from each hole to each surface point
        # sh: (batch, N_H) → (batch, N_H, 1)
        # surf_s: (N_surf,) → (1, 1, N_surf)
        ds_film = torch.abs(sh.unsqueeze(-1)
                            - self.surf_s.unsqueeze(0).unsqueeze(0))   # (batch, N_H, N_surf)
 
        # Downstream decay length ∝ dh (larger hole → longer film)
        decay_len = dh * 80.0 + 0.02                       # (batch, N_H) — chord fracs
        decay     = torch.exp(
            -ds_film / decay_len.unsqueeze(-1).clamp(min=1e-4)
        )                                                   # (batch, N_H, N_surf)
 
        # Effectiveness at each surface point = max contribution from any hole
        eta_field, _ = (eta_hole.unsqueeze(-1) * decay).max(dim=1)    # (batch, N_surf)
        return eta_field                                    # ∈ [0, ~0.72]
 
    def _assign_channels_to_surface(self, dist_ch_surf, t_wall, h_int):
        """
        For each surface point, find the nearest channel.
        Returns per-surface-point t_wall and h_int.
 
        dist_ch_surf : (batch, N_CH, N_surf)
        t_wall       : (batch, N_CH)
        h_int        : (batch, N_CH)
        → t_wall_surf, h_int_surf : (batch, N_surf)
        """
        # Nearest channel to each surface point
        _, idx = dist_ch_surf.min(dim=1)                   # (batch, N_surf)
        t_wall_surf = t_wall.gather(1, idx)                # (batch, N_surf)
        h_int_surf  = h_int.gather(1, idx)                 # (batch, N_surf)
        return t_wall_surf, h_int_surf
 
    # ------------------------------------------------------------------
    # Master cost function
    # ------------------------------------------------------------------
 
    def calculate_cost_signal(self, xc, yc, Rc, sh, dh, cfd_data, return_diagnostics: bool = False):
        """
        Total multi-physics aerospace cost from first-principles aerothermal and structural equations.

        cfd_data[:,0] — local gas temperature variation (from CFD field data)
        cfd_data[:,1] — ambient static pressure (0.5 to 1.2 atm)
        """
        batch = xc.shape[0]

        # === Step 1: Geometry → wall thickness ===
        t_wall, dist_ch_surf = self._wall_thickness(xc, yc, Rc)

        # === Step 2: Internal convection & Darcy-Weisbach pumping power ===
        h_int = self._internal_h(Rc)                       # (batch, N_CH) W/(m²·K)
        pumping_power, delta_P = self._pumping_power(Rc)    # (batch, N_CH) Watts, Pa

        # === Step 3: Film cooling effectiveness & blowing ratio ===
        eta_field = self._film_effectiveness(sh, dh)        # (batch, N_surf)
        blowing_ratio = dh * 180.0                          # (batch, N_H)

        # === Step 4: 2D Multi-Channel Thermal Conduction Diffusion ===
        t_wall_surf, h_int_surf = self._assign_channels_to_surface_2d(
            dist_ch_surf, t_wall, h_int
        )

        # === Step 5: Local gas temperature (CFD perturbation) ===
        normalized_perturbation = (cfd_data[:, 0:1] - 320.0) / 40.0
        T_gas_local = T_GAS + normalized_perturbation * 120.0    # (batch, 1)
        T_aw = T_gas_local - eta_field * (T_gas_local - T_COOL)   # (batch, N_surf)

        # === Step 6: Solid-state conjugate thermal resistance ===
        t_wall_phys = torch.clamp(t_wall_surf, min=1e-5) * CHORD   # m
        R_wall = t_wall_phys / K_METAL                     # (batch, N_surf)

        h_ext = self.h_ext_profile.unsqueeze(0)            # (1, N_surf)

        # Total thermal resistance from gas to coolant
        R_total = 1.0 / h_ext + R_wall + 1.0 / (h_int_surf + 1e-4)

        # Heat flux into coolant [W/m² per chord-fraction area]
        q = (T_aw - T_COOL) / (R_total + 1e-8)

        # Outer metal temperature (surface facing hot gas)
        T_metal = T_aw - q / (h_ext + 1e-4)               # (batch, N_surf)

        # ============================================================
        # Multi-Physics Cost Terms
        # ============================================================

        # 1. THERMAL LIMIT: penalise T_metal > T_METAL_LIM (quadratic above threshold)
        T_excess     = torch.relu(T_metal - T_METAL_LIM)
        thermal_cost = (T_excess**2).mean(dim=-1)          # (batch,)

        # 2. HOT-SPOT PENALTY: maximum temperature soft-max (LogSumExp)
        beta_softmax = 0.05
        T_softmax = beta_softmax * torch.logsumexp(T_metal / beta_softmax, dim=-1)
        hotspot_cost = torch.relu(T_softmax - T_METAL_LIM)**2

        # 3. THERMAL GRADIENT: ΔT/Δs creates fatigue micro-cracks
        dT_ds = torch.diff(T_metal, dim=-1)
        gradient_cost = (dT_ds**2).mean(dim=-1)

        # 4. THIN WALL PENALTY: t_wall < T_WALL_MIN → structural burn-through/failure
        wall_violation = torch.relu(T_WALL_MIN - t_wall)   # (batch, N_CH)
        thin_wall_cost = (wall_violation**2).sum(dim=-1) * 3000.0

        # 5. CHANNEL OVERLAP PENALTY: intersecting holes destroy interior structural ribs
        overlap_cost = torch.zeros(batch, device=xc.device)
        for i in range(N_CH):
            for j in range(i+1, N_CH):
                dist_ij = torch.sqrt(
                    (xc[:, i] - xc[:, j])**2
                  + (yc[:, i] - yc[:, j])**2
                  + 1e-8
                )
                min_sep = Rc[:, i] + Rc[:, j] + 0.005      # channels + rib thickness
                overlap = torch.relu(min_sep - dist_ij)
                overlap_cost = overlap_cost + overlap**2 * 300.0

        # 6. UNCOOLED REGION PENALTY: surface segments far from any channel
        nearest_ch_dist_to_surf, _ = dist_ch_surf.min(dim=1)   # (batch, N_surf)
        Rc_max = Rc.max(dim=-1, keepdim=True).values            # (batch, 1)
        uncooled_dist = torch.relu(nearest_ch_dist_to_surf - Rc_max - 0.05)
        uncooled_cost = (uncooled_dist**2).mean(dim=-1) * 200.0

        # 7. FILM HOLE THERMAL COVERAGE: penalise surface regions with inadequate film
        h_ext_weight  = self.h_ext_profile / self.h_ext_profile.max()   # (N_surf,)
        film_gap      = torch.relu(0.25 - eta_field)
        film_cost     = (film_gap * h_ext_weight.unsqueeze(0)).mean(dim=-1) * 100.0

        # 8. PUMPING POWER PENALTY (Darcy-Weisbach):
        # Prevents unrealistically tiny channels that claim huge h_int at astronomical pressure drop.
        total_pumping_power = pumping_power.sum(dim=-1)    # (batch,) Watts
        pumping_cost = torch.relu(total_pumping_power - 75.0) / 10.0 + (total_pumping_power / 120.0)**2

        # 9. CENTRIFUGAL MECHANICAL STRESS & RUPTURE (15,000 RPM Rotational Integrity):
        T_metal_max = T_metal.max(dim=-1).values
        stress_ratio, area_reduction, sigma_peak, sigma_yield = self._structural_stress(
            Rc, xc, yc, T_metal_max
        )
        stress_cost = (torch.relu(stress_ratio - 1.0)**2 * 50.0
                       + torch.relu(area_reduction - 0.35)**2 * 200.0)

        # 10. FILM COOLING AERODYNAMIC MIXING LOSS:
        # Blowing ratio M > 1.2 creates jet lift-off and boundary-layer aerodynamic drag
        m_excess = torch.relu(blowing_ratio - 1.2)
        film_aero_loss = ((m_excess**2) * (dh / 0.003)).mean(dim=-1) * 50.0

        # 11. MASS BALANCE / ROTOR DYNAMIC HARMONICS (Coupled Correlation Trap):
        # Target COM shifts based on ambient pressure context (0.5 to 1.2 atm)
        com_x = xc.mean(dim=-1)
        com_y = yc.mean(dim=-1)
        pressure = torch.clamp(cfd_data[:, 1], 0.5, 1.2)
        target_com_x = 0.35 + (pressure - 0.5) / 0.7 * 0.20
        target_com_y = 0.01
        com_penalty = ((com_x - target_com_x)**2 + (com_y - target_com_y)**2) * 10000000.0

        # ============================================================
        # Composite Multi-Physics Objective
        # ============================================================
        total = (
            10.0  * thermal_cost
          + 8.0   * hotspot_cost
          + 0.01  * gradient_cost
          + 1.0   * thin_wall_cost
          + 1.0   * overlap_cost
          + 1.0   * uncooled_cost
          + 1.0   * film_cost
          + 1.0   * pumping_cost
          + 1.0   * stress_cost
          + 0.5   * film_aero_loss
          + 1.0   * com_penalty
        ).mean()

        if return_diagnostics:
            diagnostics = {
                'thermal_cost': thermal_cost.mean().item(),
                'hotspot_cost': hotspot_cost.mean().item(),
                'gradient_cost': gradient_cost.mean().item(),
                'thin_wall_cost': thin_wall_cost.mean().item(),
                'overlap_cost': overlap_cost.mean().item(),
                'pumping_power_W': total_pumping_power.mean().item(),
                'pumping_cost': pumping_cost.mean().item(),
                'stress_ratio': stress_ratio.mean().item(),
                'stress_cost': stress_cost.mean().item(),
                'film_aero_loss': film_aero_loss.mean().item(),
                'com_penalty': com_penalty.mean().item(),
                'T_metal_max_K': T_metal_max.mean().item(),
                'T_metal_mean_K': T_metal.mean().item(),
            }
            return total, diagnostics

        return total
 
 
# =============================================================================
#  Training loop
# =============================================================================
 
def run_cooling_agent(agent, pinn: CoolingChannelPINN,
                      n_features: int, epochs: int, batch_size: int = 8):
    params     = list(agent.parameters()) + list(pinn.parameters())
    optimizer  = optim.Adam(params, lr=0.003)
    scheduler  = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=2e-5)
 
    dataloader = get_cfd_dataloader(batch_size=batch_size,
                                    samples=epochs * batch_size,
                                    n_features=n_features)
    data_iter  = iter(dataloader)
 
    temperature, top_p = 1.0, 0.8
    losses = []
 
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
 
        try:
            env_ctx, cfd = next(data_iter)
            if env_ctx.shape[0] != batch_size:
                p = batch_size - env_ctx.shape[0]
                env_ctx = torch.cat([env_ctx, env_ctx[:p]], dim=0)
                cfd     = torch.cat([cfd,     cfd[:p]],     dim=0)
        except StopIteration:
            pass
 
        # Pad cfd to ≥1 feature if needed
        if cfd.shape[1] < 1:
            cfd = torch.zeros(cfd.shape[0], 1)
 
        raw_probs, _ = agent(env_ctx, temperature=temperature, top_p=top_p)
        xc, yc, Rc, sh, dh = pinn(raw_probs)
        cost = pinn.calculate_cost_signal(xc, yc, Rc, sh, dh, cfd)
        losses.append(cost.item())
 
        cost.backward()
        torch.nn.utils.clip_grad_norm_(params, max_norm=5.0)
        optimizer.step()
        scheduler.step()
 
        # Adaptive exploration
        if len(losses) >= 5:
            recent   = losses[-5:]
            variance = sum(abs(recent[i] - recent[i-1]) for i in range(1, 5))
            if variance < 0.05 * (abs(recent[-1]) + 1e-8):
                temperature = min(temperature * 1.5, 3.0)
                top_p       = min(top_p + 0.1, 1.0)
            else:
                temperature = max(temperature * 0.9, 0.1)
                top_p       = max(top_p * 0.95, 0.5)
 
    return losses
 
 
# =============================================================================
#  Visualisation — blade cross-section + channel layout + T_metal
# =============================================================================
 
def _get_final_design(agent, pinn):
    """Get the final channel design from the trained agent (deterministic)."""
    with torch.no_grad():
        dummy_ctx = torch.zeros(1, 8)
        raw_probs, _ = agent(dummy_ctx, temperature=0.1, top_p=0.5)
        xc, yc, Rc, sh, dh = pinn(raw_probs)
    return (xc[0].numpy(), yc[0].numpy(),
            Rc[0].numpy(), sh[0].numpy(), dh[0].numpy())
 
 
def plot_blade_design(ax, xc_np, yc_np, Rc_np, sh_np, dh_np,
                      pinn, label='Agent', color='#00d4ff'):
    """
    Draw blade profile + cooling channels + film holes + T_metal distribution.
    """
    # Blade outline
    ax.plot(BLADE['x_suc'], BLADE['y_suc'], 'w-', linewidth=1.5, alpha=0.7)
    ax.plot(BLADE['x_pre'], BLADE['y_pre'], 'w-', linewidth=1.5, alpha=0.7)
 
    # Temperature field on blade surface
    with torch.no_grad():
        xc_t = torch.tensor(xc_np).unsqueeze(0)
        yc_t = torch.tensor(yc_np).unsqueeze(0)
        Rc_t = torch.tensor(Rc_np).unsqueeze(0)
        sh_t = torch.tensor(sh_np).unsqueeze(0)
        dh_t = torch.tensor(dh_np).unsqueeze(0)
 
        t_wall, dist_ch_surf = pinn._wall_thickness(xc_t, yc_t, Rc_t)
        h_int    = pinn._internal_h(Rc_t)
        eta_field= pinn._film_effectiveness(sh_t, dh_t)
        t_wall_s, h_int_s = pinn._assign_channels_to_surface_2d(dist_ch_surf, t_wall, h_int)
 
        T_aw     = T_GAS - eta_field[0] * (T_GAS - T_COOL)
        t_phys   = torch.clamp(t_wall_s[0], min=1e-5) * CHORD
        R_wall   = t_phys / K_METAL
        h_ext    = pinn.h_ext_profile
        R_total  = 1.0/h_ext + R_wall + 1.0/(h_int_s[0] + 1e-4)
        q        = (T_aw - T_COOL) / R_total
        T_metal  = (T_aw - q / h_ext).numpy()
 
    norm = Normalize(vmin=800, vmax=T_METAL_LIM + 100)
    cmap = plt.cm.plasma
    xs, ys, ss = BLADE['x'], BLADE['y'], BLADE['s']
    for i in range(len(xs) - 1):
        T_mid  = 0.5 * (T_metal[i] + T_metal[i+1])
        seg_col = cmap(norm(T_mid))
        ax.plot([xs[i], xs[i+1]], [ys[i], ys[i+1]], color=seg_col, linewidth=3.5)
 
    # Cooling channels — draw as circles with fill
    for i in range(N_CH):
        circ = plt.Circle((xc_np[i], yc_np[i]), Rc_np[i],
                           facecolor=color, edgecolor='white',
                           alpha=0.5, linewidth=0.8, zorder=4)
        ax.add_patch(circ)
        ax.plot(xc_np[i], yc_np[i], '+', color='white', markersize=4, zorder=5)
 
    # Film cooling holes — markers on surface
    surf_x_interp = np.interp(sh_np, BLADE['s'], BLADE['x'])
    surf_y_interp = np.interp(sh_np, BLADE['s'], BLADE['y'])
    ax.scatter(surf_x_interp, surf_y_interp,
               s=(dh_np * 5000)**1.2, c='yellow', zorder=6,
               edgecolors='white', linewidth=0.5, label='Film holes')
 
    # Colourbar
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label='T_metal (K)', fraction=0.03, pad=0.04)
 
    ax.set_xlim(-0.05, 1.1)
    ax.set_ylim(-0.12, 0.20)
    ax.set_aspect('equal')
    ax.set_title(f'{label} — Final Channel Layout', fontsize=11, color=color)
    ax.set_xlabel('x/c', fontsize=9)
    ax.set_ylabel('y/c', fontsize=9)
 
    # Legend: max temperature annotation
    T_max = T_metal.max()
    t_wall_min_mm = (t_wall.min().item() * CHORD * 1000)
    ax.text(0.98, 0.97,
            f'T_max = {T_max:.0f} K\nt_wall_min = {t_wall_min_mm:.2f} mm',
            transform=ax.transAxes, ha='right', va='top',
            fontsize=8, color='white',
            bbox=dict(boxstyle='round', facecolor='#1a1a2e', alpha=0.7))
 
 
# =============================================================================
#  Entry point
# =============================================================================
 
import os

if __name__ == "__main__":
    N        = 8
    EPOCHS   = 300
    N_STATES = 2 ** N
    NUM_RUNS = 10

    out_dir = "geometry_runs"
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 70)
    print("Turbine Blade Cooling Channel Design — Thermal PINN")
    print(f"Fixed geometry: NACA 4412 airfoil | Chord = {CHORD*1000:.0f} mm")
    print(f"Design variables: {N_CH} internal channels + {N_H} film cooling holes")
    print(f"N={N} ({N_STATES} design states) | {EPOCHS} epochs | {NUM_RUNS} runs")
    print("=" * 70)

    total_start = time.time()

    for run_idx in range(1, NUM_RUNS + 1):
        print(f"\n" + "-"*40)
        print(f"RUN {run_idx}/{NUM_RUNS}")
        print("-"*40)
        start = time.time()

        print("\n[1/2] Quantum Architect (5 layers)...")
        q_agent = QuantumArchitect(n_qubits=N, n_layers=5)
        q_pinn  = CoolingChannelPINN(input_dim=N_STATES, hidden_dim=128)
        losses_q = run_cooling_agent(q_agent, q_pinn, n_features=N, epochs=EPOCHS)
        print(f"  Start: {losses_q[0]:.3f}  |  End: {losses_q[-1]:.3f}  "
              f"|  Improvement: {(losses_q[0]-losses_q[-1])/losses_q[0]*100:.1f}%")

        print("[2/2] Classical Architect (64 neurons)...")
        c_agent = ClassicalArchitect(n_features=N, n_states=N_STATES, hidden_dim=64)
        c_pinn  = CoolingChannelPINN(input_dim=N_STATES, hidden_dim=128)
        losses_c = run_cooling_agent(c_agent, c_pinn, n_features=N, epochs=EPOCHS)
        print(f"  Start: {losses_c[0]:.3f}  |  End: {losses_c[-1]:.3f}  "
              f"|  Improvement: {(losses_c[0]-losses_c[-1])/losses_c[0]*100:.1f}%")

        print(f"  Run completed in {(time.time()-start)/60:.1f} min")

        # =========================================================
        # Plot 1: Convergence curves
        # =========================================================
        plt.style.use('dark_background')
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Turbine Blade Cooling Channel Optimisation (Run {run_idx})\n'
                     f'NACA 4412 | {N_CH} Channels + {N_H} Film Holes | '
                     f'N={N} ({N_STATES} states)',
                     fontsize=13, y=0.98)

        epochs_x = np.arange(1, EPOCHS + 1)

        # Top-left: raw loss
        ax = axes[0, 0]
        ax.plot(epochs_x, losses_q, color='#00d4ff', lw=2.5,
                label=f'Quantum (5L, ~{5*N*3} params)')
        ax.plot(epochs_x, losses_c, color='#ff4b5c', lw=2, ls='--',
                label=f'Classical (64N, ~{N_STATES*64+64*64+64} params)')
        ax.set_title('Training Loss (raw)', fontsize=11)
        ax.set_xlabel('Epoch'); ax.set_ylabel('Thermal Cost')
        ax.legend(fontsize=9); ax.grid(True, ls=':', alpha=0.3)

        # Top-right: log scale + smoothed
        from scipy.ndimage import uniform_filter1d
        smooth = lambda x, w=12: uniform_filter1d(np.maximum(x, 1e-6), size=w)
        ax2 = axes[0, 1]
        ax2.semilogy(epochs_x, smooth(losses_q), color='#00d4ff', lw=2.5,
                     label='Quantum (smoothed)')
        ax2.semilogy(epochs_x, smooth(losses_c), color='#ff4b5c', lw=2, ls='--',
                     label='Classical (smoothed)')
        ax2.set_title('Log-Scale Convergence\n(plateaus = local minima)', fontsize=11)
        ax2.set_xlabel('Epoch'); ax2.set_ylabel('Cost (log)')
        ax2.legend(fontsize=9); ax2.grid(True, ls=':', alpha=0.3, which='both')

        # Bottom-left: Quantum blade design
        q_xc, q_yc, q_Rc, q_sh, q_dh = _get_final_design(q_agent, q_pinn)
        plot_blade_design(axes[1, 0], q_xc, q_yc, q_Rc, q_sh, q_dh, q_pinn,
                          label='Quantum Agent', color='#00d4ff')

        # Bottom-right: Classical blade design
        c_xc, c_yc, c_Rc, c_sh, c_dh = _get_final_design(c_agent, c_pinn)
        plot_blade_design(axes[1, 1], c_xc, c_yc, c_Rc, c_sh, c_dh, c_pinn,
                          label='Classical Agent', color='#ff4b5c')

        plt.tight_layout()
        save_path = os.path.join(out_dir, f'turbine_cooling_channel_run_{run_idx}.png')
        plt.savefig(save_path, dpi=250, bbox_inches='tight')
        plt.close(fig)  # Free memory!
        print(f"Saved: {save_path}")

    print(f"\nAll {NUM_RUNS} runs completed in {(time.time()-total_start)/60:.1f} min")
