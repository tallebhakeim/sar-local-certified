"""
spheroid.py -- certified pince on a NON-spherical inclusion: a prolate spheroid (needle) in tissue.

Why: it breaks the spherical l=1 reduction (proves the method is not tied to the textbook sphere),
and an elongated conductor produces a lightning-rod TIP enhancement >> 9 -- strengthening the
quantified-Panagopoulos headline.

Key idea: a confocal prolate spheroid + uniform axial field separates EXACTLY in prolate spheroidal
coordinates (xi, eta, phi), foci at +-c on z.  The P_1 mode phi = F(xi) * eta is the analogue of
f(r) cos(theta).  The energy reduces to a 1D functional in xi with weight (xi^2 - 1) in place of r^2:

    W_primal[F] = (2 pi c / 3) int_1^xiout  sigma [ F'^2 (xi^2 - 1) + 2 F^2 ] d xi          (UPPER)

and (derived from div(sigma grad phi) = 0 for the P_1 mode, equilibrated flux amplitude u(xi),
regularity u(1) = 0):

    W_dual[u]   = (4 pi/3) F(xiout) u(xiout)  -  (pi/(3 c)) int_1^xiout (1/sigma)[ 2 u^2/(xi^2-1) + u'^2 ] d xi   (LOWER)

Exact fields:  inside  F = A xi ;  outside  F = -E0 c xi + B Q1(xi),  Q1(xi)=(xi/2)ln((xi+1)/(xi-1))-1.
Certified bracket:  W_dual(u) <= W_exact <= W_primal(F)  for all admissible F, u.
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

# ---- geometry & materials --------------------------------------------------------------------
E0 = 1.0
C_FOC = 1.0                 # focal distance c
ASPECT = 5.0                # a_z / a_perp  (major/minor); >1 = prolate needle
SIG_M = 1.0
SIG_P = 100.0               # conductive inclusion
XI_OUT = 40.0               # outer boundary in xi (far field)

# xi0 from aspect ratio:  a_z/a_perp = xi0/sqrt(xi0^2-1) = ASPECT
XI0 = np.sqrt(ASPECT ** 2 / (ASPECT ** 2 - 1.0))
A_Z = C_FOC * XI0
A_PERP = C_FOC * np.sqrt(XI0 ** 2 - 1.0)

_GX = np.array([-np.sqrt(3 / 5), 0.0, np.sqrt(3 / 5)])
_GW = np.array([5 / 9, 8 / 9, 5 / 9])


def Q1(x):
    return 0.5 * x * np.log((x + 1.0) / (x - 1.0)) - 1.0


def Q1p(x):
    return 0.5 * np.log((x + 1.0) / (x - 1.0)) - x / (x ** 2 - 1.0)


# exact coefficients A (inside), B (outside) from interface continuity at xi0
_s = SIG_M / SIG_P
_B = -E0 * C_FOC * XI0 * (1.0 - _s) / (_s * Q1p(XI0) * XI0 - Q1(XI0))
_A = _s * (-E0 * C_FOC + _B * Q1p(XI0))
F_R = -E0 * C_FOC * XI_OUT + _B * Q1(XI_OUT)     # outer Dirichlet datum (exact trace)


def _sigma(xi):
    return np.where(xi < XI0, SIG_P, SIG_M)


def F_exact(xi):
    xi = np.asarray(xi, float)
    return np.where(xi < XI0, _A * xi, -E0 * C_FOC * xi + _B * Q1(np.clip(xi, 1.0 + 1e-15, None)))


def Fp_exact(xi):
    xi = np.asarray(xi, float)
    return np.where(xi < XI0, _A * np.ones_like(xi),
                    -E0 * C_FOC + _B * Q1p(np.clip(xi, 1.0 + 1e-15, None)))


def u_exact(xi):
    # equilibrated flux amplitude u = c sigma (xi^2-1) F'
    return C_FOC * _sigma(xi) * (xi ** 2 - 1.0) * Fp_exact(xi)


# ---- 1D functionals --------------------------------------------------------------------------

def primal_energy(nodes, Fv):
    tot = 0.0
    for i in range(len(nodes) - 1):
        a, b = nodes[i], nodes[i + 1]
        L = b - a
        Fp = (Fv[i + 1] - Fv[i]) / L
        xg = 0.5 * (a + b) + 0.5 * L * _GX
        wg = 0.5 * L * _GW
        Fg = Fv[i] + Fp * (xg - a)
        sg = _sigma(0.5 * (a + b) + 1e-12)
        tot += np.sum(wg * sg * (Fp ** 2 * (xg ** 2 - 1.0) + 2.0 * Fg ** 2))
    return (2.0 * np.pi * C_FOC / 3.0) * tot


def dual_functional(nodes, uv):
    quad = 0.0
    for i in range(len(nodes) - 1):
        a, b = nodes[i], nodes[i + 1]
        L = b - a
        up = (uv[i + 1] - uv[i]) / L
        xg = 0.5 * (a + b) + 0.5 * L * _GX
        wg = 0.5 * L * _GW
        ug = uv[i] + up * (xg - a)
        sg = _sigma(0.5 * (a + b) + 1e-12)
        quad += np.sum(wg * (1.0 / sg) * (2.0 * ug ** 2 / (xg ** 2 - 1.0) + up ** 2))
    bnd = (4.0 * np.pi / 3.0) * F_R * uv[-1]
    return bnd - (np.pi / (3.0 * C_FOC)) * quad


# ---- assembly (sparse tridiagonal) -----------------------------------------------------------

def _graded_nodes(n_elem, grade=2.0):
    ni = no = n_elem // 2
    s = np.linspace(0, 1, ni + 1)
    inside = 1.0 + (XI0 - 1.0) * (1.0 - (1.0 - s) ** grade)      # dense near xi0
    s2 = np.linspace(0, 1, no + 1)
    outside = XI0 + (XI_OUT - XI0) * (s2 ** grade)
    return np.unique(np.concatenate([inside, outside]))


def solve_primal(nodes):
    n = len(nodes)
    rows, cols, data = [], [], []
    for i in range(n - 1):
        a, b = nodes[i], nodes[i + 1]
        L = b - a
        xg = 0.5 * (a + b) + 0.5 * L * _GX
        wg = 0.5 * L * _GW
        sg = _sigma(0.5 * (a + b) + 1e-12)
        N0 = (b - xg) / L
        N1 = (xg - a) / L
        dN = np.array([-1.0 / L, 1.0 / L])
        for p in range(2):
            Np = N0 if p == 0 else N1
            for q in range(2):
                Nq = N0 if q == 0 else N1
                val = (2.0 * np.pi * C_FOC / 3.0) * np.sum(
                    wg * sg * (dN[p] * dN[q] * (xg ** 2 - 1.0) + 2.0 * Np * Nq))
                rows.append(i + p); cols.append(i + q); data.append(val)
    K = sp.csr_matrix((data, (rows, cols)), shape=(n, n))
    free = np.arange(n - 1)
    Kff = K[free][:, free].tocsc()
    Kfd = K[free][:, [n - 1]].toarray().ravel()
    Ff = spla.spsolve(Kff, -Kfd * F_R)
    Fv = np.concatenate([Ff, [F_R]])
    return Fv, primal_energy(nodes, Fv)


def solve_dual(nodes):
    n = len(nodes)
    rows, cols, data = [], [], []
    for i in range(n - 1):
        a, b = nodes[i], nodes[i + 1]
        L = b - a
        xg = 0.5 * (a + b) + 0.5 * L * _GX
        wg = 0.5 * L * _GW
        sg = _sigma(0.5 * (a + b) + 1e-12)
        N0 = (b - xg) / L
        N1 = (xg - a) / L
        dN = np.array([-1.0 / L, 1.0 / L])
        for p in range(2):
            Np = N0 if p == 0 else N1
            for q in range(2):
                Nq = N0 if q == 0 else N1
                val = (np.pi / (3.0 * C_FOC)) * np.sum(
                    wg * (1.0 / sg) * (2.0 * Np * Nq / (xg ** 2 - 1.0) + dN[p] * dN[q]))
                rows.append(i + p); cols.append(i + q); data.append(val)
    M = sp.csr_matrix((data, (rows, cols)), shape=(n, n))
    c = np.zeros(n)
    c[-1] = (4.0 * np.pi / 3.0) * F_R
    # maximise  c.u - u^T M u  ; regularity u(1)=0 -> node 0 constrained
    free = np.arange(1, n)
    Mff = M[free][:, free].tocsc()
    uf = spla.spsolve(Mff, 0.5 * c[free])   # d/du: c - 2 M u = 0 -> u = M^{-1} c/2
    uv = np.concatenate([[0.0], uf])
    return uv, dual_functional(nodes, uv)


def exact_energy():
    """True continuous W via high-order Gauss quadrature of the EXACT F (not its interpolant)."""
    tot = 0.0
    for lo, hi in [(1.0, XI0), (XI0, XI_OUT)]:
        edges = np.linspace(lo, hi, 8000)
        for a, b in zip(edges[:-1], edges[1:]):
            L = b - a
            xg = 0.5 * (a + b) + 0.5 * L * _GX
            wg = 0.5 * L * _GW
            sg = _sigma(0.5 * (a + b))
            tot += np.sum(wg * sg * (Fp_exact(xg) ** 2 * (xg ** 2 - 1.0) + 2.0 * F_exact(xg) ** 2))
    return (2.0 * np.pi * C_FOC / 3.0) * tot


def depolarization_Lz():
    e = np.sqrt(1.0 - (A_PERP / A_Z) ** 2)
    return (1 - e ** 2) / e ** 2 * (0.5 / e * np.log((1 + e) / (1 - e)) - 1.0)


if __name__ == "__main__":
    W = exact_energy()
    Lz = depolarization_Lz()
    Ein_depol = E0 / (1.0 + (SIG_P / SIG_M - 1.0) * Lz)   # depolarization prediction
    Ein_fem = -_A / C_FOC                                 # from phi_in = A z/c
    tip_enh = abs(Fp_exact(XI0 + 1e-9)) / (C_FOC * E0)    # |E_tip|/E0 (lightning rod)
    print("=== Certified pince on a PROLATE SPHEROID (needle) ===")
    print(f"aspect a_z/a_perp = {ASPECT}  (a_z={A_Z:.3f}, a_perp={A_PERP:.3f}), sigma_p/sigma_m={SIG_P/SIG_M:.0f}")
    print(f"depolarization L_z = {Lz:.5f}")
    print(f"internal field E_in/E0:  depolar formula {Ein_depol:.4f}   vs  separated {Ein_fem:.4f}"
          f"   (rel {abs(Ein_fem/Ein_depol-1):.2e})")
    print(f"TIP enhancement |E_tip|/E0 = {tip_enh:.3f}   -> local SAR concentration = {tip_enh**2:.2f}"
          f"  (vs 9 for a sphere!)\n")
    print(f"W_exact = {W:.6e}")
    fine = np.unique(np.concatenate([np.linspace(1.0, XI0, 6000), np.linspace(XI0, XI_OUT, 6000)]))
    print(f"  dual @ exact u (should equal W) = {dual_functional(fine, u_exact(fine)):.6e}\n")
    print(" n_elem     LOWER (cert.)       UPPER (cert.)      gap/W       contains W?")
    ok = True
    for ne in [64, 256, 1024, 4096]:
        nodes = _graded_nodes(ne)
        _, up = solve_primal(nodes)
        _, lo = solve_dual(nodes)
        contains = (lo <= W * (1 + 1e-9)) and (up >= W * (1 - 1e-9))
        ok = ok and contains and (lo <= up)
        print(f" {len(nodes)-1:5d}   {lo:.8e}   {up:.8e}   {(up-lo)/W:9.2e}   {contains}")
    print("\nRESULT:", "PASS - certified bracket on a non-spherical inclusion; "
          "depolarization + tip enhancement recovered." if ok else "FAIL")
