"""
compare_anchor.py — PINN h_ext(s) vs SU2 anchor heat transfer. THE validation figure.
Reads SU2 surface CSVs (iso + adiabatic pair per pressure), reconstructs
h(s) = q_w_iso / (T_aw - T_wall) ordered by the mesh's wall-loop station, and
overlays the verifier's analytic h_ext(s). Reports L2 + peak-location error.

Usage (AFTER running SU2_CFD on the generated cfgs):
    python compare_anchor.py --p-atm 0.85
Requires: surface_turbine_anchor_<p>_{iso,adia}.csv + blade_oGrid.su2 present.
Without them it exits with a clear message (safe to run now as a dry check).
"""
import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np

ATM_TO_PA = 101325.0


def find_col(header, *cands):
    low = [h.strip().lower().replace(" ", "_") for h in header]
    for c in cands:
        if c in low:
            return low.index(c)
    raise KeyError(f"none of {cands} in columns {header}")


def load_surface(path):
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    header = rows[0]
    data = np.array([[float(v) for v in r] for r in rows[1:]])
    return header, data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p-atm", type=float, default=0.85)
    ap.add_argument("--t-wall", type=float, default=1050.0)
    ap.add_argument("--out", default="anchor_comparison.png")
    a = ap.parse_args()
    tag = f"{a.p_atm:g}".replace(".", "p")
    f_iso = f"surface_turbine_anchor_{tag}_iso.csv"
    f_adia = f"surface_turbine_anchor_{tag}_adia.csv"
    missing = [f for f in (f_iso, f_adia) if not os.path.exists(f)]
    if missing:
        print(f"Anchor CSVs not present yet (need SU2 run first): {missing}")
        print("Generate cfgs:  python make_turbine_cfg.py")
        print("Mesh:           python export_blade_mesh.py")
        print("Solve:          SU2_CFD turbine_anchor_<p>_{iso,adia}.cfg")
        return 1

    h_iso, d_iso = load_surface(f_iso)
    h_ad, d_ad = load_surface(f_adia)
    x = d_iso[:, find_col(h_iso, "x", "x_coor", "point_x")]
    qw = d_iso[:, find_col(h_iso, "heat_flux", "heatflux", "q")]
    taw = d_ad[:, find_col(h_ad, "temperature", "temp", "t")]
    h_su2 = np.abs(qw) / np.maximum(taw - a.t_wall, 1.0)

    # Arc-length station matching the PINN convention (LE -> suction -> TE -> pressure).
    ds = np.hypot(np.diff(x, append=x[0]), np.diff(
        d_iso[:, find_col(h_iso, "y", "y_coor", "point_y")], append=d_iso[0, 1]))
    s = np.cumsum(ds)
    s = s / s[-1]

    from turbine_pinn_geometry import BLADE
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    s_pinn = BLADE["s"]
    h_pinn = BLADE["h_ext"]
    # Normalize shapes (absolute levels depend on reference choices; the
    # validation target is the DISTRIBUTION: stagnation peak + suction bump).
    hn_su2 = h_su2 / h_su2.max()
    hn_pinn = h_pinn / h_pinn.max()

    l2 = float(np.sqrt(np.mean((np.interp(s_pinn, s, hn_su2) - hn_pinn) ** 2)))
    peak_su2 = float(s[np.argmax(hn_su2)])
    peak_pinn = float(s_pinn[np.argmax(hn_pinn)])
    print(f"Anchor p={a.p_atm} atm: shape-L2={l2:.4f} | "
          f"peak s: SU2={peak_su2:.3f} vs PINN={peak_pinn:.3f}")

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(s, hn_su2, color="#00d4ff", lw=2, label="SU2 anchor (RANS)")
    ax.plot(s_pinn, hn_pinn, color="#ff9f43", lw=2, ls="--", label="PINN analytic h_ext")
    ax.set_xlabel("arc-length station s (0=LE)")
    ax.set_ylabel("normalized h")
    ax.set_title(f"CFD anchor vs verifier heat-transfer distribution (p={a.p_atm} atm)")
    ax.legend()
    ax.grid(True, ls=":", alpha=0.3)
    plt.tight_layout()
    plt.savefig(a.out, dpi=200, bbox_inches="tight")
    print(f"Saved: {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
