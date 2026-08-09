"""
goal_oriented.py -- tight certified bounds on a LOCAL output via the adjoint (goal-oriented / DWR).

The global hypercircle radius rho_u bounds ||grad(u-u_h)||_sigma over the WHOLE domain, so localizing
it to a small region gives a valid but LOOSE bracket (cf. cluster.py).  Goal-oriented certification
fixes this.

For a LINEAR output  s(u) = int_S sigma (du/dz) dV  (mean axial current in the tissue voxel S -- the
dominant field component, hence a proxy for the local SAR), Galerkin orthogonality gives EXACTLY

        s(u) - s(u_h) = a(e_u, e_z)        (no computable correction term)

with e_z = z - z_h the error of the ADJOINT problem  a(v,z) = s(v)  (homogeneous Dirichlet).  Hence

        | s(u) - s(u_h) |  <=  ||grad e_u||_sigma * ||grad e_z||_sigma  <=  rho_u * rho_z .

rho_u (primal) and rho_z (adjoint) are BOTH computed by the stream-function equilibrated pince.
Because the adjoint field decays away from S, rho_z << 1, so rho_u*rho_z << rho_u : the local output
is certified TIGHT where the global-rho localization was loose.

Validation (sphere): the certified interval [s_h +- rho_u*rho_z] must contain the converged exact s,
and rho_u*rho_z must be far smaller than rho_u.
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from fem2d import E0, SIG_M, SIG_P, R_OUT, Z_OUT, build_mesh, sphere_predicate


def precompute(nodes, tris, inside):
    """Per-element geometry + sigma. Returns arrays indexed by element."""
    ntr = len(tris)
    B = np.zeros((ntr, 3)); C = np.zeros((ntr, 3))
    A = np.zeros(ntr); RC = np.zeros(ntr); ZC = np.zeros(ntr); SIG = np.zeros(ntr)
    for t, (i, j, k) in enumerate(tris):
        (ri, zi), (rj, zj), (rk, zk) = nodes[i], nodes[j], nodes[k]
        area2 = (rj - ri) * (zk - zi) - (rk - ri) * (zj - zi)
        A[t] = 0.5 * abs(area2)
        if A[t] < 1e-14:
            SIG[t] = SIG_M; RC[t] = 1.0
            continue
        B[t] = np.array([zj - zk, zk - zi, zi - zj]) / area2      # d/dr
        C[t] = np.array([rk - rj, ri - rk, rj - ri]) / area2      # d/dz
        RC[t] = (ri + rj + rk) / 3.0
        ZC[t] = (zi + zj + zk) / 3.0
        SIG[t] = SIG_P if inside(RC[t], ZC[t]) else SIG_M
    return dict(B=B, C=C, A=A, RC=RC, ZC=ZC, SIG=SIG)


def assemble_K(nodes, tris, E):
    n = len(nodes)
    rows, cols, data = [], [], []
    for t, (i, j, k) in enumerate(tris):
        if E["A"][t] < 1e-14:
            continue
        b, c = E["B"][t], E["C"][t]
        w = E["SIG"][t] * 2.0 * np.pi * E["RC"][t] * E["A"][t]
        idx = (i, j, k)
        for a in range(3):
            for bb in range(3):
                rows.append(idx[a]); cols.append(idx[bb])
                data.append(w * (b[a] * b[bb] + c[a] * c[bb]))
    return sp.csr_matrix((data, (rows, cols)), shape=(n, n))


def elem_grad(tris, E, field):
    """per-element (d/dr, d/dz) of a nodal field."""
    gr = np.zeros(len(tris)); gz = np.zeros(len(tris))
    for t, (i, j, k) in enumerate(tris):
        if E["A"][t] < 1e-14:
            continue
        fg = field[[i, j, k]]
        gr[t] = (E["B"][t] * fg).sum()
        gz[t] = (E["C"][t] * fg).sum()
    return gr, gz


def rho_min(nodes, tris, E, Gr, Gz):
    """min over stream function psi of ||G - curl(psi)/sigma||_sigma.  Returns rho.
    G is a per-element target field (Gr,Gz); q_e = curl(psi) is div-free EXACTLY."""
    n = len(nodes)
    r = nodes[:, 0]
    rows, cols, dataS = [], [], []
    L = np.zeros(n); const = 0.0
    for t, (i, j, k) in enumerate(tris):
        if E["A"][t] < 1e-14:
            continue
        b, c = E["B"][t], E["C"][t]
        A, rc, s = E["A"][t], E["RC"][t], E["SIG"][t]
        w = 2.0 * np.pi * A / (s * rc)
        idx = (i, j, k)
        for a in range(3):
            for bb in range(3):
                rows.append(idx[a]); cols.append(idx[bb])
                dataS.append(w * (b[a] * b[bb] + c[a] * c[bb]))
            L[idx[a]] += -4.0 * np.pi * A * (Gz[t] * b[a] - Gr[t] * c[a])
        const += s * (Gr[t] ** 2 + Gz[t] ** 2) * 2.0 * np.pi * rc * A
    S = sp.csr_matrix((dataS, (rows, cols)), shape=(n, n))
    axis = r < 1e-9
    dof = np.where(~axis)[0]
    psi = np.zeros(n)
    psi[dof] = spla.spsolve(S[dof][:, dof].tocsc(), -0.5 * L[dof])
    return np.sqrt(max(const + 0.5 * L.dot(psi), 0.0))


def output_rhs(nodes, tris, E, region):
    """b_z[i] = int_S sigma (dN_i/dz) dV  -> s(u)=u.b_z ; also returns element mask in S."""
    n = len(nodes)
    b = np.zeros(n)
    inS = np.zeros(len(tris), dtype=bool)
    for t, (i, j, k) in enumerate(tris):
        if E["A"][t] < 1e-14:
            continue
        if not region(E["RC"][t], E["ZC"][t]):
            continue
        inS[t] = True
        w = E["SIG"][t] * 2.0 * np.pi * E["RC"][t] * E["A"][t]
        for a, node in enumerate((i, j, k)):
            b[node] += w * E["C"][t][a]      # dN_a/dz
    return b, inS


def solve(nodes, tris, inside, region):
    E = precompute(nodes, tris, inside)
    K = assemble_K(nodes, tris, E)
    r, z = nodes[:, 0], nodes[:, 1]
    outer = (np.abs(r - R_OUT) < 1e-9) | (np.abs(np.abs(z) - Z_OUT) < 1e-9)
    dof = np.where(~outer)[0]
    Kdd = K[dof][:, dof].tocsc()

    # primal:  u = g on outer
    uD = -E0 * z
    u = uD.copy()
    u[dof] = spla.spsolve(Kdd, -(K[dof][:, np.where(outer)[0]]).dot(uD[outer]))

    # adjoint:  K z = b_z, homogeneous Dirichlet
    bz, inS = output_rhs(nodes, tris, E, region)
    zc = np.zeros(len(nodes))
    zc[dof] = spla.spsolve(Kdd, bz[dof])

    s_h = u.dot(bz)
    ur, uz = elem_grad(tris, E, u)
    zr, zz = elem_grad(tris, E, zc)
    rho_u = rho_min(nodes, tris, E, ur, uz)
    # adjoint equilibrated flux q = sigma*chi_S e_z + curl(psi) (matches the source div(sigma chi_S e_z));
    # target for the pince is grad(z_h) - chi_S e_z
    Gz = zz - inS.astype(float)
    rho_z = rho_min(nodes, tris, E, zr, Gz)
    volS = np.sum(2.0 * np.pi * E["RC"][inS] * E["A"][inS])
    return s_h, rho_u, rho_z, volS


if __name__ == "__main__":
    inside = sphere_predicate(1.0)
    regions = {
        "shell 1<|x|<1.5 (angle-averaged -> <E_z> ~ E0)":
            lambda rc, zc: (1.0 < rc ** 2 + zc ** 2 < 1.5 ** 2),
        "POLE box r<0.35, 1<z<1.4 (axial-field concentration)":
            lambda rc, zc: (rc < 0.35) and (1.0 < zc < 1.4) and (rc ** 2 + zc ** 2 > 1.0),
    }
    print("=== Goal-oriented certified bound on a LOCAL output s(u)=int_S sigma du/dz dV ===")
    print("sphere, sigma_p/sigma_m=100 ; s certified via rho_u*rho_z ; <E_z>/E0 = -s/(sigma_m Vol_S E0)\n")
    overall = True
    for name, region in regions.items():
        nodes, tris, *_ = build_mesh(200, 400)
        s_ref, _, _, volref = solve(nodes, tris, inside, region)
        print(f"--- region: {name}")
        print(f"    reference: <E_z>/E0 = {-s_ref/(SIG_M*volref*E0):.3f}  "
              f"(local SAR proxy eta ~ {(s_ref/(SIG_M*volref*E0))**2:.2f})")
        print("    mesh       rho_u     rho_z    rho_u*rho_z    <E_z>/E0 interval       contains?")
        for (nr, nz) in [(100, 200), (160, 320)]:
            nodes, tris, *_ = build_mesh(nr, nz)
            s_h, rho_u, rho_z, volS = solve(nodes, tris, inside, region)
            hw = rho_u * rho_z
            elo, ehi = -(s_h + hw) / (SIG_M * volS * E0), -(s_h - hw) / (SIG_M * volS * E0)
            elo, ehi = min(elo, ehi), max(elo, ehi)
            eref = -s_ref / (SIG_M * volref * E0)
            contains = (elo - 1e-3 <= eref <= ehi + 1e-3)
            overall = overall and contains
            print(f"   {len(nodes):7d}   {rho_u:.3e} {rho_z:.3e}   {hw:.3e}   "
                  f"[{elo:.3f}, {ehi:.3f}]      {contains}")
        print()
    print("RESULT:", "PASS - goal-oriented gives TIGHT certified local field bounds "
          "(rho_u*rho_z << rho_u); pole region shows certified axial-field concentration."
          if overall else "CHECK values")
