"""
export_blade_mesh.py — Algebraic O-grid around the NACA 4412 blade, SU2 format.
STAGE 1 of CFD anchoring: needs NO solver. Produces a RANS-ready 2D mesh whose
surface station ordering matches the PINN's arc-length coordinate `s`, so SU2
surface heat flux maps 1:1 onto the verifier's h_ext(s) profile.

Mesh: structured O-grid, quadrilateral elements, hyperbolic-tangent wall
clustering (y+~1 capable), farfield ~20 chords. Markers: "airfoil" (wall),
"farfield". Output: ASCII .su2 (NDIME=2).

Usage:
    python export_blade_mesh.py            # writes blade_oGrid.su2 (defaults)
    python export_blade_mesh.py --n-surf 256 --n-normal 96 --farfield 20
"""
import argparse
import math
import numpy as np

CHORD = 0.050  # m (must match turbine_pinn_geometry.CHORD)


def naca4412(n_surf):
    """Upper+lower surface loop, LE -> suction -> TE -> pressure -> LE."""
    xi = np.linspace(0.0, 1.0, n_surf // 2)
    t = 0.12
    y_t = 5 * t * (0.2969 * np.sqrt(xi + 1e-12) - 0.1260 * xi
                   - 0.3516 * xi**2 + 0.2843 * xi**3 - 0.1015 * xi**4)
    m, p = 0.04, 0.40
    y_c = np.where(xi < p, m / p**2 * (2 * p * xi - xi**2),
                   m / (1 - p)**2 * ((1 - 2 * p) + 2 * p * xi - xi**2))
    dy_c = np.where(xi < p, 2 * m / p**2 * (p - xi),
                    2 * m / (1 - p)**2 * (p - xi))
    theta = np.arctan(dy_c)
    x_suc = xi - y_t * np.sin(theta)
    y_suc = y_c + y_t * np.cos(theta)
    x_pre = xi + y_t * np.sin(theta)
    y_pre = y_c - y_t * np.cos(theta)
    # Closed loop starting/ending at TE keeps the O-grid periodic and clean.
    x_loop = np.concatenate([x_suc[::-1], x_pre[1:]])
    y_loop = np.concatenate([y_suc[::-1], y_pre[1:]])
    return x_loop, y_loop


def build_o_grid(n_surf=256, n_normal=96, farfield_chords=20.0,
                 first_ds_chord=2e-5, growth=1.12):
    """
    Outward marching: each surface node marches along its outward normal with
    geometrically growing steps; outer ring is blended to a circle of radius
    `farfield_chords` so the farfield marker is clean.
    Returns xs, ys in chord units, shape (n_normal+1, n_loop).
    """
    xs, ys = naca4412(n_surf)
    n_loop = len(xs)
    # Outward normals via loop tangent (2D cross product).
    tx = np.gradient(xs)
    ty = np.gradient(ys)
    mag = np.hypot(tx, ty) + 1e-12
    nx, ny = ty / mag, -tx / mag
    # Fix orientation: must point AWAY from blade centroid.
    cx, cy = xs.mean(), ys.mean()
    sgn = np.sign((xs - cx) * nx + (ys - cy) * ny)
    nx, ny = nx * sgn, ny * sgn

    # Smooth normals (circular moving average): raw gradient normals chatter
    # at the leading-edge curvature singularity and fold the marching front.
    k = 5
    ker = np.ones(k) / k
    nx = np.convolve(np.concatenate([nx[-k:], nx, nx[:k]]), ker, mode="same")[k:k + n_loop]
    ny = np.convolve(np.concatenate([ny[-k:], ny, ny[:k]]), ker, mode="same")[k:k + n_loop]
    mag = np.hypot(nx, ny) + 1e-12
    nx, ny = nx / mag, ny / mag

    ds = first_ds_chord * growth ** np.arange(n_normal + 1)
    dist = np.concatenate([[0.0], np.cumsum(ds)])[:n_normal + 1]
    dist = dist / dist[-1] * farfield_chords  # pin outer extent to farfield
    # Marching DIRECTION blends normal -> radial: pure normal marching tangles
    # where neighbor rays converge; pure radial marching from a star-shaped
    # loop is a diffeomorphism (rays never cross). Blend factor grows with ring.
    ux = (xs - cx) / (np.hypot(xs - cx, ys - cy) + 1e-12)
    uy = (ys - cy) / (np.hypot(xs - cx, ys - cy) + 1e-12)
    w = (np.arange(n_normal + 1) / n_normal) ** 2  # 0 wall -> 1 farfield
    dx = (1 - w)[None, :] * nx[:, None] + w[None, :] * ux[:, None]
    dy = (1 - w)[None, :] * ny[:, None] + w[None, :] * uy[:, None]
    dm = np.hypot(dx, dy) + 1e-12
    px = xs[:, None] + dx / dm * dist[None, :]
    py = ys[:, None] + dy / dm * dist[None, :]
    return px.T, py.T  # (rings, loop) with ring 0 = wall


def write_su2(path, xs, ys):
    """Minimal ASCII .su2: NPOIN points, quad elements, airfoil+farfield markers."""
    rings, loop = xs.shape
    npoin = rings * loop
    quads = []
    for j in range(rings - 1):
        for i in range(loop):
            i2 = (i + 1) % loop
            # CCW winding (inner_i -> outer_i -> outer_{i+1} -> inner_{i+1}):
            # the wall loop runs CCW and marching goes outward, so this order
            # yields positive-Jacobian cells. (The naive inner-first order
            # produces inside-out cells — caught by the self-check below.)
            quads.append((j * loop + i, (j + 1) * loop + i,
                          (j + 1) * loop + i2, j * loop + i2))
    # Self-check: shoelace signed areas must be (almost) all positive.
    pts = np.stack([xs.ravel(), ys.ravel()], axis=-1)
    neg = 0
    for q in quads:
        p = pts[list(q)]
        e = np.roll(p, -1, axis=0)
        if float((p[:, 0] * e[:, 1] - e[:, 0] * p[:, 1]).sum() / 2.0) <= 0:
            neg += 1
    rate = neg / len(quads)
    print(f"mesh self-check: {neg}/{len(quads)} inverted cells ({rate:.4f})")
    assert rate < 0.01, "mesh folded: tune smoothing/growth before running SU2"
    with open(path, "w") as f:
        f.write("% NACA4412 O-grid for turbine SU2 anchor (chord units)\n")
        f.write("NDIME= 2\n")
        f.write(f"NPOIN= {npoin}\n")
        for j in range(rings):
            for i in range(loop):
                f.write(f"{xs[j, i]:.10f} {ys[j, i]:.10f} 0\n")
        f.write(f"NELEM= {len(quads)}\n")
        for q in quads:
            f.write(f"9 {' '.join(map(str, q))}\n")  # 9 = quadrilateral
        f.write("NMARK= 2\n")
        f.write("MARKER_TAG= airfoil\n")
        f.write(f"MARKER_ELEMS= {loop}\n")
        for i in range(loop):
            i2 = (i + 1) % loop
            f.write(f"3 {i} {i2}\n")  # 3 = line element
        f.write("MARKER_TAG= farfield\n")
        f.write(f"MARKER_ELEMS= {loop}\n")
        base = (rings - 1) * loop
        for i in range(loop):
            i2 = (i + 1) % loop
            f.write(f"3 {base + i} {base + i2}\n")
    print(f"Wrote {path}: {npoin} points, {len(quads)} quads, loop={loop}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-surf", type=int, default=256)
    ap.add_argument("--n-normal", type=int, default=96)
    ap.add_argument("--farfield", type=float, default=20.0)
    ap.add_argument("--out", default="blade_oGrid.su2")
    a = ap.parse_args()
    xs, ys = build_o_grid(a.n_surf, a.n_normal, a.farfield)
    write_su2(a.out, xs, ys)
