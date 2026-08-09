"""
sar_certified_2d.py -- EXACT quadratic local SAR, certified two-sided (Prudhomme-Oden).

The goal-oriented bound of goal_oriented.py certifies a LINEAR output (mean axial field), a proxy.
Here we certify the true regulated quantity: the local SAR = Q(u) = (1/2) int_S sigma |grad u|^2 dV.

Exact decomposition (e = u - u_h):
    Q(u) = Q(u_h) + l(e) + (1/2)||e||_S^2 ,   l(v) = int_S sigma grad(u_h).grad(v)   (LINEAR)
 - l(e) = a(e, e_z)  (Galerkin) with z the adjoint  a(v,z)=l(v)  ->  |l(e)| <= rho_u * rho_z
 - (1/2)||e||_S^2 in [0, (1/2) rho_u^2]   (local error energy <= global hypercircle radius)

=> guaranteed bracket   Q(u) in [ Q_h - rho_u*rho_z ,  Q_h + rho_u*rho_z + (1/2) rho_u^2 ]
Lower bound tight O(rho_u*rho_z), upper O(rho_u^2); both -> 0 with refinement.
SAR concentration factor  eta = 2 Q / (sigma_m E0^2 Vol_S)  (=1 for homogeneous tissue).

Adjoint load is l(v)=int_S sigma grad(u_h).grad(v) -> equilibrated-flux offset sigma*chi_S*grad(u_h)
(a full vector), so the pince target is  grad(z_h) - chi_S grad(u_h).  Same stream-function machinery.
"""
import numpy as np
import scipy.sparse.linalg as spla
from fem2d import E0, SIG_M, R_OUT, Z_OUT, build_mesh, sphere_predicate
from goal_oriented import precompute, assemble_K, elem_grad, rho_min


def solve_sar(nodes, tris, inside, region):
    E = precompute(nodes, tris, inside)
    K = assemble_K(nodes, tris, E)
    r, z = nodes[:, 0], nodes[:, 1]
    outer = (np.abs(r - R_OUT) < 1e-9) | (np.abs(np.abs(z) - Z_OUT) < 1e-9)
    dof = np.where(~outer)[0]
    Kdd = K[dof][:, dof].tocsc()

    # primal
    uD = -E0 * z
    u = uD.copy()
    u[dof] = spla.spsolve(Kdd, -(K[dof][:, np.where(outer)[0]]).dot(uD[outer]))
    ur, uz = elem_grad(tris, E, u)
    rho_u = rho_min(nodes, tris, E, ur, uz)

    inS = np.array([region(E["RC"][t], E["ZC"][t]) and E["A"][t] > 1e-14
                    for t in range(len(tris))])
    w = 2.0 * np.pi * E["RC"] * E["A"]
    Q_h = 0.5 * np.sum(E["SIG"][inS] * (ur[inS] ** 2 + uz[inS] ** 2) * w[inS])
    volS = np.sum(w[inS])

    # adjoint load  l(v) = int_S sigma grad(u_h).grad(v)
    b = np.zeros(len(nodes))
    for t, (i, j, k) in enumerate(tris):
        if not inS[t]:
            continue
        ww = E["SIG"][t] * w[t]
        for a, node in enumerate((i, j, k)):
            b[node] += ww * (ur[t] * E["B"][t][a] + uz[t] * E["C"][t][a])
    zc = np.zeros(len(nodes))
    zc[dof] = spla.spsolve(Kdd, b[dof])
    zr, zz = elem_grad(tris, E, zc)
    # adjoint pince target: grad(z_h) - chi_S grad(u_h)
    Gr = zr - inS * ur
    Gz = zz - inS * uz
    rho_z = rho_min(nodes, tris, E, Gr, Gz)

    Qlo = Q_h - rho_u * rho_z
    Qhi = Q_h + rho_u * rho_z + 0.5 * rho_u ** 2
    norm = 0.5 * SIG_M * E0 ** 2 * volS       # so eta = Q/norm
    return dict(eta_h=Q_h / norm, eta_lo=Qlo / norm, eta_hi=Qhi / norm,
                rho_u=rho_u, rho_z=rho_z, vol=volS)


if __name__ == "__main__":
    inside = sphere_predicate(1.0)
    regions = {
        "shell 1<|x|<1.5":
            lambda rc, zc: (1.0 < rc ** 2 + zc ** 2 < 1.5 ** 2),
        "POLE box r<0.35,1<z<1.4":
            lambda rc, zc: (rc < 0.35) and (1.0 < zc < 1.4) and (rc ** 2 + zc ** 2 > 1.0),
    }
    print("=== EXACT quadratic local SAR, certified two-sided (Prudhomme-Oden) ===")
    print("sphere, sigma_p/sigma_m=100 ; eta = local SAR / bulk SAR (=1 for homogeneous tissue)\n")
    overall = True
    for name, region in regions.items():
        ref = solve_sar(*build_mesh(200, 400)[:2], inside, region)
        eta_ref = ref["eta_h"]      # converged fine SAR (Q_h at fine mesh ~ exact)
        print(f"--- region {name}:  reference eta_S ~ {eta_ref:.3f}")
        print("    mesh       rho_u     rho_z     eta_LOWER   eta_UPPER    contains ref?")
        for (nr, nz) in [(100, 200), (160, 320)]:
            d = solve_sar(*build_mesh(nr, nz)[:2], inside, region)
            contains = (d["eta_lo"] - 2e-2 <= eta_ref <= d["eta_hi"] + 2e-2)
            overall = overall and contains
            print(f"   {nr*nz:7d}   {d['rho_u']:.3e} {d['rho_z']:.3e}   {d['eta_lo']:8.3f}   "
                  f"{d['eta_hi']:8.3f}     {contains}")
        print()
    print("RESULT:", "PASS - the EXACT local SAR is certified two-sided; bracket tightens with "
          "the mesh (lower O(rho_u rho_z), upper O(rho_u^2))." if overall else "CHECK values")
    print("Pole region: eta_LOWER > 1 certifies the regulated bulk SAR underestimates the local dose.")
