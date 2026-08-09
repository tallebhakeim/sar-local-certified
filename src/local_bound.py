"""
local_bound.py -- certified interval on the LOCAL (region-averaged) SAR, not just the global power.

This attacks the crux of the article: can the *local* dosimetric quantity be certified, or does
only the global energy admit guaranteed bounds?  Answer here: YES, via hypercircle localisation.

Prager-Synge gives, for the compatible gradient g_c = grad(phi_c) and the equilibrated flux
q_e = sigma grad(phi_e) (both from certified_bracket.py), the IDENTITY

    || grad(u) - g_c ||^2_sigma  +  || grad(u) - q_e/sigma ||^2_sigma  =  || g_c - q_e/sigma ||^2_sigma  =: rho^2

so  || grad(u) - g_c ||_sigma <= rho   (rho fully computable, no exact solution used).

For ANY sub-region S (here a tissue shell [a,b] adjacent to the particle -- the "voxel"):

    | ||grad u||_{sigma,S} - ||g_c||_{sigma,S} |  <=  ||grad u - g_c||_{sigma,S}  <=  rho

    =>   E_S(true) = int_S sigma|grad u|^2  in  [ (||g_c||_{S} - rho)_+^2 , (||g_c||_{S} + rho)^2 ]

a GUARANTEED interval on the local absorbed power / region-averaged SAR.  rho -> 0 as the mesh
refines, so the interval tightens.  (Width is set by the *global* radius rho; an adjoint localised
to S -- true goal-oriented DWR -- tightens it further; that is the v2 optimisation, but the crux
"is the local quantity certifiable at all?" is answered here.)
"""
import numpy as np
from certified_bracket import (E0, A_RAD, R_OUT, SIG_M, BETA, _GX, _GW, _sigma,
                               _elem_nodes, solve_primal, solve_dual,
                               f_exact, fp_exact)

SHELL = (A_RAD, 1.5 * A_RAD)   # adjacent tissue voxel (outside the particle)


def _fem_amp(nodes, vals, r):
    """Piecewise-linear amplitude value and derivative at r (r inside a single element)."""
    i = np.clip(np.searchsorted(nodes, r) - 1, 0, len(nodes) - 2)
    r0, r1 = nodes[i], nodes[i + 1]
    v0, v1 = vals[i], vals[i + 1]
    L = r1 - r0
    val = v0 + (v1 - v0) * (r - r0) / L
    der = (v1 - v0) / L
    return val, der


def _integrate(nodes, integrand, r_lo=None, r_hi=None):
    """(4pi/3)*A + (8pi/3)*B style handled by caller; here generic sum of integrand(rg)*wg."""
    tot = 0.0
    for i in range(len(nodes) - 1):
        r0, r1 = nodes[i], nodes[i + 1]
        if r_lo is not None and r1 <= r_lo:
            continue
        if r_hi is not None and r0 >= r_hi:
            continue
        lo = max(r0, r_lo) if r_lo is not None else r0
        hi = min(r1, r_hi) if r_hi is not None else r1
        L = hi - lo
        rg = 0.5 * (lo + hi) + 0.5 * L * _GX
        wg = 0.5 * L * _GW
        tot += np.sum(wg * integrand(rg, i))
    return tot


def rho_hypercircle(nodes, fvals, hvals):
    """rho^2 = ||g_c - q_e/sigma||^2_sigma, radial l=1 reduction."""
    def integ(rg, i):
        r0, r1 = nodes[i], nodes[i + 1]
        L = r1 - r0
        fc = fvals[i] + (fvals[i + 1] - fvals[i]) * (rg - r0) / L
        fcp = (fvals[i + 1] - fvals[i]) / L
        h = hvals[i] + (hvals[i + 1] - hvals[i]) * (rg - r0) / L
        hp = (hvals[i + 1] - hvals[i]) / L
        k = -(h + rg * hp / 2.0)
        sg = _sigma(0.5 * (r0 + r1) + 1e-12)
        # radial diff (cos): fcp - h/sigma ; angular diff (sin): (fc/r) - (-k/sigma)?  see note
        rad = fcp - h / sg
        ang = (fc / rg) + k / sg          # g_c,theta amp = -fc/r ; (q_e/sig),theta amp = k/sig
        # ||.||^2_sigma with weights (4pi/3) radial, (8pi/3) angular, times r^2 dr
        return sg * (rad ** 2 * rg ** 2 * (4.0 * np.pi / 3.0)
                     + ang ** 2 * rg ** 2 * (8.0 * np.pi / 3.0)) / (4.0 * np.pi / 3.0)
    # factor bookkeeping: we folded weights inside; integrate integrand*wg then multiply (4pi/3)
    val = _integrate(nodes, integ)
    return (4.0 * np.pi / 3.0) * val


def shell_energy_field(nodes, fvals, shell):
    """||g_c||^2_{sigma,S} = int_S sigma|grad phi_c|^2 over shell S."""
    def integ(rg, i):
        r0, r1 = nodes[i], nodes[i + 1]
        L = r1 - r0
        fc = fvals[i] + (fvals[i + 1] - fvals[i]) * (rg - r0) / L
        fcp = (fvals[i + 1] - fvals[i]) / L
        sg = _sigma(0.5 * (r0 + r1) + 1e-12)
        return sg * (fcp ** 2 * rg ** 2 * (4.0 * np.pi / 3.0)
                     + (fc / rg) ** 2 * rg ** 2 * (8.0 * np.pi / 3.0)) / (4.0 * np.pi / 3.0)
    val = _integrate(nodes, integ, shell[0], shell[1])
    return (4.0 * np.pi / 3.0) * val


def shell_energy_exact(shell):
    fine = np.linspace(shell[0], shell[1], 20000)
    r = 0.5 * (fine[:-1] + fine[1:])
    dr = np.diff(fine)
    sg = _sigma(r)
    fc = f_exact(r)
    fcp = fp_exact(r)
    integ = sg * (fcp ** 2 * r ** 2 * (4.0 * np.pi / 3.0)
                  + (fc / r) ** 2 * r ** 2 * (8.0 * np.pi / 3.0))
    return np.sum(integ * dr)


def eta_shell_exact(shell):
    """Region-averaged local SAR concentration factor over the shell (exact)."""
    E_S = shell_energy_exact(shell)
    vol = (4.0 * np.pi / 3.0) * (shell[1] ** 3 - shell[0] ** 3)
    return E_S / (SIG_M * E0 ** 2 * vol)


if __name__ == "__main__":
    E_true = shell_energy_exact(SHELL)
    vol = (4.0 * np.pi / 3.0) * (SHELL[1] ** 3 - SHELL[0] ** 3)
    norm = SIG_M * E0 ** 2 * vol
    eta_ex = E_true / norm
    print("=== Certified interval on LOCAL (shell-averaged) SAR ===")
    print(f"shell (tissue voxel) = [{SHELL[0]:.2f}, {SHELL[1]:.2f}] a   beta={BETA:.4f}")
    print(f"eta_shell exact (region-averaged concentration factor) = {eta_ex:.4f}")
    print(f"  (pole point-value would be |1+2beta|^2 = {(1+2*BETA)**2:.3f}; shell mixes pole+equator)")
    print(f"E_shell exact = {E_true:.6e}\n")
    print(" n_elem      rho        LOWER (cert.)      UPPER (cert.)    eta_lo    eta_hi   contains?")
    ok = True
    for ne in [64, 256, 1024, 4096]:
        nodes = _elem_nodes(ne)
        fvals, _ = solve_primal(nodes)
        hvals, _ = solve_dual(nodes)
        rho = np.sqrt(max(rho_hypercircle(nodes, fvals, hvals), 0.0))
        Ec_S = np.sqrt(max(shell_energy_field(nodes, fvals, SHELL), 0.0))
        lo = max(Ec_S - rho, 0.0) ** 2
        hi = (Ec_S + rho) ** 2
        contains = (lo <= E_true + 1e-6) and (hi >= E_true - 1e-6)
        ok = ok and contains
        print(f" {len(nodes)-1:5d}  {rho:.3e}  {lo:.8e}   {hi:.8e}  {lo/norm:7.4f}  "
              f"{hi/norm:7.4f}   {contains}")
    print("\nRESULT:", "PASS - local SAR is certifiable; interval brackets the exact value "
          "and tightens with the mesh." if ok else "FAIL")
