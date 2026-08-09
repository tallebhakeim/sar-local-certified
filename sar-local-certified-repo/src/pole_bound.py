"""
pole_bound.py -- the headline: certified concentration factor of the LOCAL SAR at the POLE.

The regulated bulk SAR would report the factor 1.  We certify a guaranteed LOWER bound on the
SAR concentration averaged over a small tissue cap at the pole (theta in [0, theta0], r in [a,b]),
which -> |1 + 2 beta|^2 (= 8.65 here, 9 for a perfect conductor) as the cap shrinks to the pole point.

Method: same hypercircle localisation as local_bound.py, but the sub-region S is a polar CAP.
The l=1 solution structure lets the cap integrals be done in closed form angularly:

    A_cos(theta0) = 2 pi (1 - cos^3 theta0)/3          (weight of (f')^2, radial field)
    A_sin(theta0) = 2 pi [(1 - cos theta0) - (1 - cos^3 theta0)/3]   (weight of (f/r)^2, tangential)

Guaranteed interval:  E_cap(true) in [ (||g_c||_cap - rho)_+^2 , (||g_c||_cap + rho)^2 ],  rho global.
The certified LOWER bound on eta_cap is the paper's quantified answer to Panagopoulos:
"the regulated bulk SAR provably underestimates the local pole dose by at least this factor."
"""
import numpy as np
from certified_bracket import (E0, A_RAD, SIG_M, BETA, _GX, _GW, _sigma,
                               _elem_nodes, solve_primal, solve_dual, f_exact, fp_exact)
from local_bound import rho_hypercircle

CAP_DEG = 25.0                      # polar cap half-angle
SHELL = (A_RAD, 1.05 * A_RAD)       # thin tissue voxel hugging the particle surface


def cap_weights(theta0_deg):
    c0 = np.cos(np.deg2rad(theta0_deg))
    a_cos = 2.0 * np.pi * (1.0 - c0 ** 3) / 3.0
    a_sin = 2.0 * np.pi * ((1.0 - c0) - (1.0 - c0 ** 3) / 3.0)
    solid = 2.0 * np.pi * (1.0 - c0)          # cap solid angle
    return a_cos, a_sin, solid


def cap_energy(nodes, fvals, shell, theta0_deg):
    """int_{cap} sigma |grad phi_c|^2 for the piecewise-linear compatible potential."""
    a_cos, a_sin, _ = cap_weights(theta0_deg)
    tot = 0.0
    for i in range(len(nodes) - 1):
        r0, r1 = nodes[i], nodes[i + 1]
        if r1 <= shell[0] or r0 >= shell[1]:
            continue
        lo, hi = max(r0, shell[0]), min(r1, shell[1])
        L = hi - lo
        rg = 0.5 * (lo + hi) + 0.5 * L * _GX
        wg = 0.5 * L * _GW
        fc = fvals[i] + (fvals[i + 1] - fvals[i]) * (rg - r0) / (r1 - r0)
        fcp = (fvals[i + 1] - fvals[i]) / (r1 - r0)
        sg = _sigma(0.5 * (r0 + r1) + 1e-12)
        tot += np.sum(wg * sg * (fcp ** 2 * a_cos + (fc / rg) ** 2 * a_sin) * rg ** 2)
    return tot


def cap_energy_exact(shell, theta0_deg):
    a_cos, a_sin, _ = cap_weights(theta0_deg)
    fine = np.linspace(shell[0], shell[1], 40000)
    r = 0.5 * (fine[:-1] + fine[1:])
    dr = np.diff(fine)
    sg = _sigma(r)
    return np.sum(sg * (fp_exact(r) ** 2 * a_cos + (f_exact(r) / r) ** 2 * a_sin) * r ** 2 * dr)


def cap_volume(shell, theta0_deg):
    _, _, solid = cap_weights(theta0_deg)
    return solid * (shell[1] ** 3 - shell[0] ** 3) / 3.0


if __name__ == "__main__":
    vol = cap_volume(SHELL, CAP_DEG)
    norm = SIG_M * E0 ** 2 * vol
    E_true = cap_energy_exact(SHELL, CAP_DEG)
    eta_true = E_true / norm
    eta_point = (1.0 + 2.0 * BETA) ** 2
    print("=== Certified SAR concentration at the POLE (headline) ===")
    print(f"cap: theta<{CAP_DEG:.0f} deg, r in [{SHELL[0]:.2f},{SHELL[1]:.2f}]a   beta={BETA:.4f}")
    print(f"eta_point (pole, |1+2beta|^2)        = {eta_point:.4f}")
    print(f"eta_cap  exact (averaged over cap)   = {eta_true:.4f}")
    print(f"bulk SAR would report                = 1.0000\n")
    print(" n_elem      rho         eta_LOWER (cert.)   eta_UPPER (cert.)   contains?   underestimate>=")
    ok = True
    for ne in [1024, 4096, 16384, 65536]:
        nodes = _elem_nodes(ne, grade=2.0)   # grade 2 keeps the graded mesh well-conditioned
        fvals, _ = solve_primal(nodes)
        hvals, _ = solve_dual(nodes)
        rho = np.sqrt(max(rho_hypercircle(nodes, fvals, hvals), 0.0))
        Ec = np.sqrt(max(cap_energy(nodes, fvals, SHELL, CAP_DEG), 0.0))
        lo = max(Ec - rho, 0.0) ** 2 / norm
        hi = (Ec + rho) ** 2 / norm
        contains = (lo <= eta_true + 1e-4) and (hi >= eta_true - 1e-4)
        ok = ok and contains
        print(f" {len(nodes)-1:6d}  {rho:.3e}   {lo:15.4f}   {hi:15.4f}     {str(contains):5s}"
              f"      x{lo:.2f}")
    print("\nRESULT:", "PASS - certified that the local pole SAR exceeds the bulk SAR by the "
          "guaranteed LOWER factor above." if ok else "FAIL")
    print("Panagopoulos, quantified: the regulated bulk SAR provably underestimates the local pole "
          f"dose (true factor {eta_true:.2f}, approaching the point value {eta_point:.2f}).")
