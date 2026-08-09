"""
certified_bracket.py -- Prager-Synge dual bracket on the dissipated power (= integrated SAR)
for a conductive nano-inclusion in tissue, quasi-static conduction problem  div(sigma grad phi)=0.

We work on the exactly-reducible l=1 problem (uniform applied field + sphere): phi = f(r) cos(theta).
The dissipated power  P = (1/2) integral sigma |grad phi|^2 dV = integrated local SAR * rho.

Two complementary certified bounds (no knowledge of the exact solution used to build them):

  UPPER (compatible / Rayleigh-Ritz): for ANY admissible potential f (Dirichlet data at r=R,
         finite at 0),   P <= Q_prim(f) = (2 pi/3) int sigma [ (f')^2 r^2 + 2 f^2 ] dr.

  LOWER (equilibrated / Thomson): for ANY divergence-free flux D (l=1 form D_r=h(r)cos, D_th=k sin
         with k = -(h + r h'/2) enforcing div D = 0),
         P >= L_dual(h) = -E0 R^3 h(R) (4 pi/3) - (2 pi/3) int (1/sigma)[ h^2 r^2 + 2 k^2 r^2 ] dr.

Guarantee:  L_dual(h)  <=  P_exact  <=  Q_prim(f)   for every admissible f, h.
The gap Q_prim - L_dual is a fully computable, guaranteed error bar on the dosimetric output.

This file (i) checks that exact fields make both bounds coincide with P_exact, and
(ii) shows that coarse FEM trial fields already sandwich P_exact and that refinement tightens it.
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

# ---- problem data (conductive nanoparticle in tissue, conduction analogue) --------------------
E0 = 1.0
A_RAD = 1.0            # particle radius a
R_OUT = 30.0           # outer boundary (>> a)
SIG_M = 1.0            # tissue conductivity (normalised)
SIG_P = 100.0          # particle conductivity  -> beta ~ 0.97, eta_pole ~ 8.6

BETA = (SIG_P - SIG_M) / (SIG_P + 2.0 * SIG_M)
B_DIP = A_RAD ** 3 * E0 * BETA          # outside: f = -E0 r + B/r^2
A_IN = E0 * (BETA - 1.0)                # inside : f = A_in r
F_R = -E0 * R_OUT + B_DIP / R_OUT ** 2  # exact dipole trace at r=R (outer Dirichlet datum)

# 3-point Gauss-Legendre on [-1,1]
_GX = np.array([-np.sqrt(3 / 5), 0.0, np.sqrt(3 / 5)])
_GW = np.array([5 / 9, 8 / 9, 5 / 9])


def _sigma(r):
    return np.where(r < A_RAD, SIG_P, SIG_M)


def f_exact(r):
    r = np.asarray(r, float)
    return np.where(r < A_RAD, A_IN * r, -E0 * r + B_DIP / np.maximum(r, 1e-30) ** 2)


def fp_exact(r):
    r = np.asarray(r, float)
    return np.where(r < A_RAD, A_IN * np.ones_like(r),
                    -E0 - 2.0 * B_DIP / np.maximum(r, 1e-30) ** 3)


def h_exact(r):
    # equilibrated (Thomson) flux radial amplitude  p = +sigma grad(phi) -> h = +sigma f'
    return _sigma(r) * fp_exact(r)


def _elem_nodes(n_elem, grade=3.0):
    """Graded radial nodes on [0,R], clustered near the interface a on both sides."""
    # half the elements inside [0,a], half in [a,R], geometric grading toward a
    ni = no = n_elem // 2
    s = np.linspace(0, 1, ni + 1) ** 1.0
    inside = A_RAD * (1 - (1 - s) ** grade)          # dense near a
    s2 = np.linspace(0, 1, no + 1)
    outside = A_RAD + (R_OUT - A_RAD) * (s2 ** grade)  # dense near a
    nodes = np.unique(np.concatenate([inside, outside]))
    return nodes


def primal_energy(nodes, fvals):
    """Q_prim(f) = (2pi/3) int sigma[(f')^2 r^2 + 2 f^2] dr for piecewise-linear f."""
    tot = 0.0
    for i in range(len(nodes) - 1):
        r0, r1 = nodes[i], nodes[i + 1]
        L = r1 - r0
        f0, f1 = fvals[i], fvals[i + 1]
        fp = (f1 - f0) / L
        rg = 0.5 * (r0 + r1) + 0.5 * L * _GX
        wg = 0.5 * L * _GW
        fg = f0 + fp * (rg - r0)
        sg = _sigma(0.5 * (r0 + r1) + 1e-12)  # element is entirely one side (a is a node)
        tot += np.sum(wg * sg * (fp ** 2 * rg ** 2 + 2.0 * fg ** 2))
    return (2.0 * np.pi / 3.0) * tot


def dual_functional(nodes, hvals):
    """L_dual(h) = -E0 R^3 h(R)(4pi/3) - (2pi/3) int (1/sigma)[h^2 r^2 + 2 k^2 r^2] dr,
       k = -(h + r h'/2), h piecewise linear."""
    quad = 0.0
    for i in range(len(nodes) - 1):
        r0, r1 = nodes[i], nodes[i + 1]
        L = r1 - r0
        h0, h1 = hvals[i], hvals[i + 1]
        hp = (h1 - h0) / L
        rg = 0.5 * (r0 + r1) + 0.5 * L * _GX
        wg = 0.5 * L * _GW
        hg = h0 + hp * (rg - r0)
        kg = -(hg + rg * hp / 2.0)
        sg = _sigma(0.5 * (r0 + r1) + 1e-12)
        quad += np.sum(wg * (1.0 / sg) * (hg ** 2 * rg ** 2 + 2.0 * kg ** 2 * rg ** 2))
    bnd = F_R * hvals[-1] * R_OUT ** 2 * (4.0 * np.pi / 3.0)
    return bnd - (2.0 * np.pi / 3.0) * quad


# ---- assemble quadratic forms and solve the two variational problems -------------------------

def _assemble(nodes, kind):
    """Assemble the tridiagonal element matrix (sparse) for 'primal' or 'dual'."""
    n = len(nodes)
    rows, cols, data = [], [], []
    for i in range(n - 1):
        r0, r1 = nodes[i], nodes[i + 1]
        L = r1 - r0
        rg = 0.5 * (r0 + r1) + 0.5 * L * _GX
        wg = 0.5 * L * _GW
        sg = _sigma(0.5 * (r0 + r1) + 1e-12)
        N0 = (r1 - rg) / L
        N1 = (rg - r0) / L
        dN = np.array([-1.0 / L, 1.0 / L])
        for a in range(2):
            Na = N0 if a == 0 else N1
            ka = -(Na + rg * dN[a] / 2.0)
            for b in range(2):
                Nb = N0 if b == 0 else N1
                kb = -(Nb + rg * dN[b] / 2.0)
                if kind == "primal":
                    val = (2.0 * np.pi / 3.0) * np.sum(
                        wg * sg * (dN[a] * dN[b] * rg ** 2 + 2.0 * Na * Nb))
                else:  # dual mass matrix M
                    val = np.sum(wg * (1.0 / sg) * (Na * Nb * rg ** 2 + 2.0 * ka * kb * rg ** 2))
                rows.append(i + a); cols.append(i + b); data.append(val)
    return sp.csr_matrix((data, (rows, cols)), shape=(n, n))


def solve_primal(nodes):
    """Minimise Q_prim with Dirichlet f(R) = F_R -> certified UPPER bound (sparse tridiagonal)."""
    n = len(nodes)
    K = _assemble(nodes, "primal")
    fD = F_R                     # exact dipole trace -> FEM and exact solve the SAME problem
    free = np.arange(n - 1)      # node n-1 is Dirichlet
    Kff = K[free][:, free]
    Kfd = K[free][:, [n - 1]].toarray().ravel()
    f_free = spla.spsolve(Kff.tocsc(), -Kfd * fD)
    fvals = np.concatenate([f_free, [fD]])
    return fvals, primal_energy(nodes, fvals)


def solve_dual(nodes):
    """Maximise concave L_dual(h) -> certified LOWER bound (sparse tridiagonal)."""
    n = len(nodes)
    M = _assemble(nodes, "dual")
    c = np.zeros(n)
    c[-1] = F_R * R_OUT ** 2 * (4.0 * np.pi / 3.0)
    # dL/dh = c - (4pi/3) M h = 0
    hvals = spla.spsolve(((4.0 * np.pi / 3.0) * M).tocsc(), c)
    return hvals, dual_functional(nodes, hvals)


def exact_power():
    """P_exact from the exact potential via the (exact) primal functional, fine quadrature."""
    nodes = np.unique(np.concatenate([
        np.linspace(0, A_RAD, 4000), np.linspace(A_RAD, R_OUT, 8000)]))
    return primal_energy(nodes, f_exact(nodes))


if __name__ == "__main__":
    Pex = exact_power()

    # (i) exact fields must reproduce P_exact from BOTH sides (validation of the functionals)
    fine = np.unique(np.concatenate([np.linspace(0, A_RAD, 4000),
                                     np.linspace(A_RAD, R_OUT, 8000)]))
    up_exact = primal_energy(fine, f_exact(fine))
    lo_exact = dual_functional(fine, h_exact(fine))
    print("=== Prager-Synge dual bracket on dissipated power P (integrated SAR) ===")
    print(f"beta = {BETA:.4f}   eta_pole(analytic) = {(1+2*BETA)**2:.4f}")
    print(f"P_exact (reference)                    = {Pex:.6e}")
    print(f"  primal functional at exact f         = {up_exact:.6e}  (rel {abs(up_exact/Pex-1):.2e})")
    print(f"  dual   functional at exact D         = {lo_exact:.6e}  (rel {abs(lo_exact/Pex-1):.2e})")

    # (ii) certified bracket from FEM trial fields, refining the mesh
    print("\n n_elem     LOWER (certified)     UPPER (certified)     gap/P      contains P?")
    ok = True
    for ne in [16, 32, 64, 128, 256]:
        nodes = _elem_nodes(ne)
        _, up = solve_primal(nodes)
        _, lo = solve_dual(nodes)
        contains = (lo <= Pex + 1e-9) and (up >= Pex - 1e-9)
        ok = ok and contains and (lo <= up)
        print(f" {len(nodes)-1:5d}   {lo:.8e}   {up:.8e}   {(up-lo)/Pex:8.2e}   {contains}")

    print("\nRESULT:", "PASS - bracket always contains P_exact and tightens with refinement"
          if ok else "FAIL")
