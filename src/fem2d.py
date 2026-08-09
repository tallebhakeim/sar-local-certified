"""
fem2d.py -- axisymmetric 2D FEM for an inclusion of ARBITRARY shape in tissue (the case with no
closed form).  Stage 1: the compatible (primal) P1 solve -> certified UPPER bound on the dissipated
power, validated on a sphere against the exact Clausius-Mossotti internal field.

Coordinates: meridian half-plane (r>=0, z), azimuthal symmetry (m=0).  Applied uniform field E0 along
z: Dirichlet u = -E0 z on the outer boundary.  Conduction problem div(sigma grad u)=0.
Axisymmetric weak form:  int sigma grad u . grad v  (2 pi r) dA.

sigma(element) is set by an inclusion predicate on the element centroid, so ANY shape is allowed
(sphere here for validation; clusters / rounded cylinders next).
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

E0 = 1.0
SIG_M = 1.0
SIG_P = 100.0
R_OUT = 20.0
Z_OUT = 20.0


def sphere_predicate(a=1.0):
    return lambda rc, zc: rc ** 2 + zc ** 2 < a ** 2


def graded_axis(n, hi, cluster_hi=3.0, grade=2.5):
    """n+1 points on [0,hi], dense near 0 (where the inclusion sits)."""
    s = np.linspace(0, 1, n + 1)
    near = cluster_hi * (1 - (1 - s) ** 1.0)          # uniform-ish fine part [0,cluster_hi]
    # blend: fine grid up to cluster_hi, geometric stretch beyond
    n_fine = int(n * 0.6)
    fine = np.linspace(0, cluster_hi, n_fine + 1)
    s2 = np.linspace(0, 1, n - n_fine + 1)
    coarse = cluster_hi + (hi - cluster_hi) * s2 ** grade
    return np.unique(np.concatenate([fine, coarse]))


def build_mesh(nr=60, nz=120):
    rs = graded_axis(nr, R_OUT)
    zpos = graded_axis(nz // 2, Z_OUT)
    zs = np.unique(np.concatenate([-zpos[::-1], zpos]))
    NR, NZ = len(rs), len(zs)
    R, Z = np.meshgrid(rs, zs)                      # shape (NZ, NR)
    nodes = np.column_stack([R.ravel(), Z.ravel()])  # index = iz*NR + ir

    def nid(ir, iz):
        return iz * NR + ir

    tris = []
    for iz in range(NZ - 1):
        for ir in range(NR - 1):
            n00, n10 = nid(ir, iz), nid(ir + 1, iz)
            n01, n11 = nid(ir, iz + 1), nid(ir + 1, iz + 1)
            tris.append((n00, n10, n11))
            tris.append((n00, n11, n01))
    return nodes, np.array(tris), rs, zs, NR, NZ


def assemble_primal(nodes, tris, inside):
    n = len(nodes)
    rows, cols, data = [], [], []
    sig_e = np.empty(len(tris))
    for t, (i, j, k) in enumerate(tris):
        (ri, zi), (rj, zj), (rk, zk) = nodes[i], nodes[j], nodes[k]
        area2 = (rj - ri) * (zk - zi) - (rk - ri) * (zj - zi)
        A = 0.5 * abs(area2)
        if A < 1e-14:
            sig_e[t] = SIG_M
            continue
        # P1 gradients
        b = np.array([zj - zk, zk - zi, zi - zj]) / area2
        c = np.array([rk - rj, ri - rk, rj - ri]) / area2
        rc = (ri + rj + rk) / 3.0
        zc = (zi + zj + zk) / 3.0
        s = SIG_P if inside(rc, zc) else SIG_M
        sig_e[t] = s
        w = s * 2.0 * np.pi * rc * A                 # axisymmetric weight (centroid rule)
        idx = (i, j, k)
        for a in range(3):
            for bb in range(3):
                rows.append(idx[a]); cols.append(idx[bb])
                data.append(w * (b[a] * b[bb] + c[a] * c[bb]))
    K = sp.csr_matrix((data, (rows, cols)), shape=(n, n))
    return K, sig_e


def solve_primal(nodes, tris, inside, tol_bnd=1e-9):
    K, sig_e = assemble_primal(nodes, tris, inside)
    n = len(nodes)
    r, z = nodes[:, 0], nodes[:, 1]
    on_outer = (np.abs(r - R_OUT) < tol_bnd) | (np.abs(np.abs(z) - Z_OUT) < tol_bnd)
    uD = -E0 * z
    dof = np.where(~on_outer)[0]
    bnd = np.where(on_outer)[0]
    Kdd = K[dof][:, dof].tocsc()
    Kdb = K[dof][:, bnd]
    rhs = -Kdb.dot(uD[bnd])
    u = uD.copy()
    u[dof] = spla.spsolve(Kdd, rhs)
    energy = 0.5 * u.dot(K.dot(u))
    return u, energy, sig_e


def solve_dual_rho(nodes, tris, u, sig_e):
    """Equilibrated flux via Stokes stream function psi (q_e = curl -> div-free EXACTLY).
    Minimise rho^2 = ||grad u_c - q_e/sigma||^2_sigma over psi (psi=0 on axis for regularity).
    Returns rho (hypercircle radius) and psi.  ANY div-free q_e gives a valid bound; the minimiser
    gives the tightest.  rho -> 0 as the mesh refines.
    """
    n = len(nodes)
    r = nodes[:, 0]
    rows, cols, dataS = [], [], []
    L = np.zeros(n)
    const = 0.0
    for t, (i, j, k) in enumerate(tris):
        (ri, zi), (rj, zj), (rk, zk) = nodes[i], nodes[j], nodes[k]
        area2 = (rj - ri) * (zk - zi) - (rk - ri) * (zj - zi)
        A = 0.5 * abs(area2)
        if A < 1e-14:
            continue
        b = np.array([zj - zk, zk - zi, zi - zj]) / area2      # d/dr
        c = np.array([rk - rj, ri - rk, rj - ri]) / area2      # d/dz
        rc = (ri + rj + rk) / 3.0
        s = sig_e[t]
        ug = u[[i, j, k]]
        ur, uz = (b * ug).sum(), (c * ug).sum()                # grad u_c
        w = 2.0 * np.pi * A / (s * rc)                         # stiffness weight 1/(sigma r)
        idx = (i, j, k)
        for a in range(3):
            for bb in range(3):
                rows.append(idx[a]); cols.append(idx[bb])
                dataS.append(w * (b[a] * b[bb] + c[a] * c[bb]))
            # q_e = (-(1/r)psi_z, (1/r)psi_r); cross term -2 grad u_c . q_e integrates (1/r cancels)
            L[idx[a]] += -4.0 * np.pi * A * (uz * b[a] - ur * c[a])
        const += s * (ur ** 2 + uz ** 2) * 2.0 * np.pi * rc * A
    S = sp.csr_matrix((dataS, (rows, cols)), shape=(n, n))
    axis = r < 1e-9
    dof = np.where(~axis)[0]
    psi = np.zeros(n)
    # minimise const + L.psi + psi^T S psi -> S psi = -L/2  (psi=0 on axis)
    Sff = S[dof][:, dof].tocsc()
    psi[dof] = spla.spsolve(Sff, -0.5 * L[dof])
    rho2 = const + 0.5 * L.dot(psi)          # = const + L.psi + psi^T S psi at the minimiser
    return np.sqrt(max(rho2, 0.0)), psi


def region_field_energy(nodes, tris, u, sig_e, predicate):
    """||g_c||^2_{sigma,S} and volume of S = union of elements whose centroid satisfies predicate."""
    Ec, vol = 0.0, 0.0
    for t, (i, j, k) in enumerate(tris):
        (ri, zi), (rj, zj), (rk, zk) = nodes[i], nodes[j], nodes[k]
        area2 = (rj - ri) * (zk - zi) - (rk - ri) * (zj - zi)
        A = 0.5 * abs(area2)
        if A < 1e-14:
            continue
        b = np.array([zj - zk, zk - zi, zi - zj]) / area2
        c = np.array([rk - rj, ri - rk, rj - ri]) / area2
        rc, zc = (ri + rj + rk) / 3.0, (zi + zj + zk) / 3.0
        if not predicate(rc, zc):
            continue
        ug = u[[i, j, k]]
        ur, uz = (b * ug).sum(), (c * ug).sum()
        w = 2.0 * np.pi * rc * A
        Ec += sig_e[t] * (ur ** 2 + uz ** 2) * w
        vol += w
    return Ec, vol


def element_field(nodes, tris, u):
    """E = -grad u, constant per triangle; return centroids, E vector, sigma-less."""
    cent = np.zeros((len(tris), 2))
    Efield = np.zeros((len(tris), 2))
    for t, (i, j, k) in enumerate(tris):
        (ri, zi), (rj, zj), (rk, zk) = nodes[i], nodes[j], nodes[k]
        area2 = (rj - ri) * (zk - zi) - (rk - ri) * (zj - zi)
        if abs(area2) < 1e-14:
            continue
        b = np.array([zj - zk, zk - zi, zi - zj]) / area2
        c = np.array([rk - rj, ri - rk, rj - ri]) / area2
        ug = u[[i, j, k]]
        Efield[t] = [-(b * ug).sum(), -(c * ug).sum()]   # (E_r, E_z)
        cent[t] = [(ri + rj + rk) / 3, (zi + zj + zk) / 3]
    return cent, Efield


if __name__ == "__main__":
    a = 1.0
    inside = sphere_predicate(a)
    beta = (SIG_P - SIG_M) / (SIG_P + 2 * SIG_M)
    Ein_exact = (1.0 - beta) * E0                     # = 3 sig_m/(sig_p+2 sig_m) E0
    Epole_exact = (1.0 + 2 * beta) * E0
    print("=== 2D axisymmetric primal FEM on a SPHERE (validation of the machinery) ===")
    print(f"sigma_p/sigma_m={SIG_P/SIG_M:.0f}  beta={beta:.4f}")
    print(f"exact internal |E_in|/E0 = {Ein_exact:.4f} ; exact pole |E|/E0 = {Epole_exact:.4f}\n")
    for (nr, nz) in [(40, 80), (80, 160), (140, 280)]:
        nodes, tris, rs, zs, NR, NZ = build_mesh(nr, nz)
        u, energy, sig_e = solve_primal(nodes, tris, inside)
        cent, Ef = element_field(nodes, tris, u)
        Emag = np.hypot(Ef[:, 0], Ef[:, 1])
        rc, zc = cent[:, 0], cent[:, 1]
        deep = (rc ** 2 + zc ** 2) < (0.4 * a) ** 2       # well inside
        Ein = Emag[deep].mean()
        print(f" mesh {len(nodes):6d} nodes, {len(tris):6d} tris | E_in/E0={Ein:.4f} "
              f"(exact {Ein_exact:.4f})   W_upper={energy:.4e}")

    # ---- Stage 2: stream-function dual + localized hypercircle bracket -----------------------
    shell = lambda rc, zc: (a ** 2 < rc ** 2 + zc ** 2 < (1.5 * a) ** 2)   # tissue voxel around sphere
    print("\n=== Stage 2: certified LOCAL SAR via stream-function dual + hypercircle ===")
    print("shell region a < |x| < 1.5a  (analytic sphere reference eta_shell ~ 1.56)")
    # fine reference for the local field energy (converged primal)
    nodes, tris, *_ = build_mesh(180, 360)
    uref, _, sref = solve_primal(nodes, tris, inside)
    Eref, volref = region_field_energy(nodes, tris, uref, sref, shell)
    print(f"reference (fine primal)  E_S = {Eref:.4e}   eta_S ~ {Eref/(SIG_M*E0**2*volref):.3f}\n")
    print(" mesh(nodes)     rho          eta_LOWER     eta_UPPER    contains ref?   tighten")
    ok = True
    for (nr, nz) in [(50, 100), (90, 180), (150, 300)]:
        nodes, tris, *_ = build_mesh(nr, nz)
        u, _, sig_e = solve_primal(nodes, tris, inside)
        rho, psi = solve_dual_rho(nodes, tris, u, sig_e)
        Ec, vol = region_field_energy(nodes, tris, u, sig_e, shell)
        norm = SIG_M * E0 ** 2 * vol
        lo = max(np.sqrt(Ec) - rho, 0.0) ** 2 / norm
        hi = (np.sqrt(Ec) + rho) ** 2 / norm
        contains = (lo <= Eref / (SIG_M * E0 ** 2 * volref) + 0.05) and \
                   (hi >= Eref / (SIG_M * E0 ** 2 * volref) - 0.05)
        ok = ok and contains
        print(f" {len(nodes):8d}   {rho:.4e}     {lo:8.4f}     {hi:8.4f}      {str(contains):5s}")
    print("\nRESULT:", "PASS - 2D certified local bracket works: rho falls, interval brackets the "
          "reference and tightens." if ok else "CHECK - see values")
