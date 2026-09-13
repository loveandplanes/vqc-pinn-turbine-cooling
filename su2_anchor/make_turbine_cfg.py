"""
make_turbine_cfg.py — SU2 2D RANS case files for the blade anchor.
Generates a matched pair per operating point:
  * isothermal wall (T_wall fixed)  -> surface heat flux q_w(s)
  * adiabatic wall                   -> recovery/adiabatic wall temperature T_aw(s)
Together: h(s) = q_w_iso(s) / (T_aw(s) - T_wall). Directly comparable to the
PINN verifier's h_ext(s) profile. Compressible SA-RANS, engine-representative
freestream; pressures span the project's 0.5–1.2 atm envelope.

Usage:
    python make_turbine_cfg.py            # writes cfgs for 3 anchor pressures
Requires (at run time, NOT now): SU2_CFD + blade_oGrid.su2 from
export_blade_mesh.py. Run:  SU2_CFD turbine_anchor_<p>.cfg
"""
import os

# Engine-representative cascade-ish inflow (2D blade section).
MACH = 0.55
AOA_DEG = 0.0
T_FREESTREAM_K = 1650.0       # hot gas, matches PINN T_GAS
T_WALL_K = 1050.0             # isothermal wall (below T_METAL_LIM=1100 K)
REYNOLDS_PER_M = 4.0e6        # per metre; x REYNOLDS_LENGTH=chord -> engine Re
CHORD_M = 0.050
ANCHOR_PRESSURES_ATM = [0.6, 0.85, 1.1]
ATM_TO_PA = 101325.0

CFG_TEMPLATE = """% Turbine blade SU2 anchor: {wall_desc} | p = {p_atm} atm
SOLVER= RANS
KIND_TURB_MODEL= SPALART_ALLMARAS
MATH_PROBLEM= DIRECT
RESTART_SOL= NO
MACH_NUMBER= {mach}
AOA= {aoa}
FREESTREAM_TEMPERATURE= {t_inf}
FREESTREAM_PRESSURE= {p_pa}
REYNOLDS_NUMBER= {reynolds:.4e}
REYNOLDS_LENGTH= {chord}
REF_ORIGIN_MOMENT_X= 0.25
REF_ORIGIN_MOMENT_Y= 0.0
REF_AREA= {chord}
REF_LENGTH= {chord}
MARKER_HEATFLUX= ( airfoil, 0.0 )
MARKER_ISOTHERMAL= ( airfoil, {t_wall} )
MARKER_FAR= ( farfield )
MARKER_MONITORING= ( airfoil )
MARKER_PLOTTING= ( airfoil )
MARKER_ANALYZE= ( airfoil )
NUM_METHOD_GRAD= GREEN_GAUSS
CFL_NUMBER= 5.0
CFL_ADAPT= YES
CFL_ADAPT_PARAM= ( 1.5, 0.5, 10.0, 100.0 )
LINEAR_SOLVER= FGMRES
LINEAR_SOLVER_PREC= LU_SGS
LINEAR_SOLVER_ERROR= 1E-4
LINEAR_SOLVER_ITER= 10
CONV_NUM_METHOD_MEANFLOW= ROE
CONV_NUM_METHOD_TURB= SCALAR_UPWIND
MUSCL_FLOW= YES
SLOPE_LIMITER_FLOW= VENKATAKRISHNAN
VENKAT_LIMITER_COEFF= 0.05
TIME_DISCRE_FLOW= EULER_IMPLICIT
TIME_DISCRE_TURB= EULER_IMPLICIT
RELAXATION_FACTOR_FLOW= 1.0
RELAXATION_FACTOR_TURB= 1.0
ITER= 4000
CONV_RESIDUAL_MINVAL= -8
CONV_STARTITER= 10
CONV_CAUCHY_ELEMS= 100
CONV_CAUCHY_EPS= 1E-6
MESH_FILENAME= blade_oGrid.su2
MESH_FORMAT= SU2
TABULAR_FORMAT= CSV
OUTPUT_FILES= (RESTART, PARAVIEW, SURFACE_CSV)
SURFACE_OUTPUT= (COEFF_PRESSURE, HEAT_FLUX)
WRT_CSV_SOL= YES
SCREEN_OUTPUT= (INNER_ITER, RMS_DENSITY, RMS_ENERGY, RMS_NU_TILDE, LIFT, DRAG)
HISTORY_OUTPUT= (INNER_ITER, RMS_DENSITY, RMS_ENERGY, RMS_NU_TILDE)
"""

# NOTE: SU2 applies the LAST applicable wall marker per zone; the isothermal
# run and the adiabatic run are separate cfgs (heatflux 0.0 with no isothermal
# marker = adiabatic). We emit both explicitly to avoid marker ambiguity.


def emit(p_atm, isothermal, outdir="."):
    p_pa = p_atm * ATM_TO_PA
    tag = f"{p_atm:g}".replace(".", "p")
    mode = "iso" if isothermal else "adia"
    wall_desc = f"isothermal T={T_WALL_K}K" if isothermal else "adiabatic"
    lines = []
    for raw in CFG_TEMPLATE.format(
            wall_desc=wall_desc, p_atm=p_atm, mach=MACH, aoa=AOA_DEG,
            t_inf=T_FREESTREAM_K, p_pa=p_pa, reynolds=REYNOLDS_PER_M * CHORD_M,
            chord=CHORD_M,
            t_wall=T_WALL_K if isothermal else 0.0).splitlines(keepends=True):
        if (not isothermal) and raw.startswith("MARKER_ISOTHERMAL"):
            continue  # adiabatic run: pure zero-heatflux wall
        lines.append(raw)
    path = os.path.join(outdir, f"turbine_anchor_{tag}_{mode}.cfg")
    with open(path, "w") as f:
        f.writelines(lines)
    print(f"Wrote {path}")
    return path


if __name__ == "__main__":
    for p in ANCHOR_PRESSURES_ATM:
        emit(p, isothermal=True)
        emit(p, isothermal=False)
    print("Run each with:  SU2_CFD <cfg>   (needs solver + blade_oGrid.su2)")
