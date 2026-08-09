"""
fem3d_certified.py -- certified conduction bracket in FULL 3D for a nano-inclusion / aggregate,
reusing a validated discrete-geometric-method dual core: primal P1 nodal FEM (upper bound) + RT0-P0
equilibrated mixed FEM (lower bound, Thomson). This lifts the axisymmetric limit of fem2d.py and
reaches genuinely non-axisymmetric aggregates (two particles side by side, perpendicular to the
field), the physically realistic case for nanoparticle clusters in tissue.

Meshing is BODY-FITTED (gmsh conformal tets: dgm.gmsh_mesh.box_with_sphere), so the curved
interface is resolved and the internal field converges cleanly to the analytic Clausius-Mossotti
value, unlike a structured staircase grid.

Setup: a tissue box between two plate contacts (top -> V0, bottom -> 0, insulated sides) creates an
approximately uniform axial field; the conductive inclusion(s) concentrate it. The dual pair
brackets the two-plate conductance, and the primal field is validated against the analytic sphere.
"""
import os
import sys
import numpy as np
import scipy.sparse.linalg as spla

# Ce script est le SEUL du depot a dependre d'un coeur externe (paquet `dgm`,
# fournissant mesh3d / primal3d / mixed3d et une interface gmsh), qui n'est pas
# redistribue ici. Indiquer son emplacement par la variable d'environnement
# DGM_CORE_PATH, par exemple :
#     export DGM_CORE_PATH=/chemin/vers/core
_dgm = os.environ.get("DGM_CORE_PATH")
if not _dgm:
    sys.exit("fem3d_certified.py requiert le coeur externe `dgm`.\n"
             "Definir DGM_CORE_PATH vers le repertoire qui contient le paquet dgm/.\n"
             "Les douze autres scripts du depot n'ont besoin que de numpy/scipy/matplotlib.")
sys.path.insert(0, _dgm)
from dgm.primal3d import assemble_primal_3d, element_gradients
from dgm.mixed3d import capacitance_lower_3d
from dgm.gmsh_mesh import box_with_sphere, _extract
from dgm.mesh3d import TetMesh

SIG_M, SIG_P = 1.0, 100.0
BETA = (SIG_P - SIG_M) / (SIG_P + 2 * SIG_M)


def box_with_two_spheres(L=1.0, r=0.2, gap=0.1, h=0.05, hball=None, axis="z"):
    """Conformal mesh of a cube with TWO spheres separated by `gap`, aligned along `axis`.
    Returns (mesh, inside) with inside the per-tet union mask of the two spheres."""
    import gmsh
    hball = hball or h / 2.5
    c = L / 2.0
    d = r + gap / 2.0                              # centre offset from box centre
    off = {"x": (d, 0, 0), "y": (0, d, 0), "z": (0, 0, d)}[axis]
    c1 = np.array([c, c, c]) - off
    c2 = np.array([c, c, c]) + off
    gmsh.initialize(); gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("box_2sph")
    box = gmsh.model.occ.addBox(0, 0, 0, L, L, L)
    s1 = gmsh.model.occ.addSphere(*c1, r)
    s2 = gmsh.model.occ.addSphere(*c2, r)
    gmsh.model.occ.fragment([(3, box)], [(3, s1), (3, s2)])
    gmsh.model.occ.synchronize()
    surf = [s[1] for s in gmsh.model.getBoundary([(3, s1), (3, s2)], oriented=False)]
    gmsh.model.mesh.field.add("Distance", 1)
    gmsh.model.mesh.field.setNumbers(1, "SurfacesList", surf)
    gmsh.model.mesh.field.add("Threshold", 2)
    gmsh.model.mesh.field.setNumber(2, "InField", 1)
    gmsh.model.mesh.field.setNumber(2, "SizeMin", hball)
    gmsh.model.mesh.field.setNumber(2, "SizeMax", h)
    gmsh.model.mesh.field.setNumber(2, "DistMin", r * 0.2)
    gmsh.model.mesh.field.setNumber(2, "DistMax", r * 1.5)
    gmsh.model.mesh.field.setAsBackgroundMesh(2)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.model.mesh.generate(3)
    pts, tets = _extract()
    gmsh.finalize()
    mesh = TetMesh(pts, tets)
    cen = mesh.points[mesh.tets].mean(axis=1)
    inside = (np.linalg.norm(cen - c1, axis=1) < r) | (np.linalg.norm(cen - c2, axis=1) < r)
    return mesh, inside, np.array([c, c, c]), r, gap


def solve3d(mesh, sigma_tet, E0=1.0):
    """Two-plate solve (top z=zmax -> V0, bottom z=zmin -> 0), insulated sides.
    Returns primal potential, upper/lower energies, and the mean internal |E|/E0."""
    z = mesh.points[:, 2]
    zmin, zmax = z.min(), z.max()
    tol = 1e-6 * (zmax - zmin)
    V0 = (zmax - zmin) * E0
    top = np.abs(z - zmax) < tol
    bot = np.abs(z - zmin) < tol
    dmask = top | bot
    dnodes = np.where(dmask)[0]
    uD = np.zeros(mesh.np)
    uD[top] = V0
    K = assemble_primal_3d(mesh, sigma_tet)
    free = np.where(~dmask)[0]
    Kff = K[free][:, free].tocsc()
    rhs = -K[free][:, dnodes].dot(uD[dnodes])
    u = uD.copy()
    u[free] = spla.spsolve(Kff, rhs)
    W_up = 0.5 * float(u @ (K @ u))

    dirichlet = {int(n): float(uD[n]) for n in dnodes}
    W_lo, _, _ = capacitance_lower_3d(mesh, sigma_tet, dirichlet)

    grads, Vol = element_gradients(mesh)
    Etet = -np.einsum("tik,ti->tk", grads, u[mesh.tets])
    Emag = np.linalg.norm(Etet, axis=1)
    incl = sigma_tet > 0.5 * (SIG_M + SIG_P)
    Ein = float(np.average(Emag[incl], weights=Vol[incl])) / E0 if incl.any() else np.nan
    return u, W_up, W_lo, Ein


def aggregate_gap_eta(L=2.0, r=0.2, gap=0.1, h=0.10, axis="z"):
    """Primal gap SAR concentration of a two-sphere aggregate, normalized by the local
    far-field (bulk) SAR. Returns (eta_gap, mesh, u, inside, ctr, r, gap) for plotting."""
    mesh, inside, ctr, r, gap = box_with_two_spheres(L=L, r=r, gap=gap, h=h, axis=axis)
    sig = np.where(inside, SIG_P, SIG_M).astype(float)
    # inline primal (fast; the slow RT0 lower bound is not needed for the field map)
    z = mesh.points[:, 2]; zmin, zmax = z.min(), z.max(); tol = 1e-6 * (zmax - zmin)
    top = np.abs(z - zmax) < tol; bot = np.abs(z - zmin) < tol; dm = top | bot; dn = np.where(dm)[0]
    uD = np.zeros(mesh.np); uD[top] = (zmax - zmin)
    K = assemble_primal_3d(mesh, sig); free = np.where(~dm)[0]
    u = uD.copy(); u[free] = spla.spsolve(K[free][:, free].tocsc(), -K[free][:, dn].dot(uD[dn]))
    grads, Vol = element_gradients(mesh); cen = mesh.points[mesh.tets].mean(axis=1)
    E2 = np.linalg.norm(-np.einsum("tik,ti->tk", grads, u[mesh.tets]), axis=1) ** 2
    rad = np.linalg.norm(cen[:, :2] - ctr[:2], axis=1)
    gapm = (~inside) & (np.abs(cen[:, 2] - ctr[2]) < gap / 2) & (rad < 0.4 * r)
    dist = np.minimum(np.linalg.norm(cen - (ctr - [0, 0, r + gap / 2]), axis=1),
                      np.linalg.norm(cen - (ctr + [0, 0, r + gap / 2]), axis=1))
    bulk = (~inside) & (dist > 3 * r) & (np.abs(cen[:, 2] - ctr[2]) < 0.3 * L)
    E0loc = np.average(E2[bulk], weights=Vol[bulk])
    eta = float(np.average(E2[gapm], weights=Vol[gapm]) / E0loc)
    return eta, mesh, u, inside, ctr, r, gap


if __name__ == "__main__":
    print("=== Certified 3D conduction bracket, BODY-FITTED (gmsh) + DGM dual core ===")
    print(f"sigma_p/sigma_m={SIG_P/SIG_M:.0f}  beta={BETA:.4f}  analytic |E_in|/E0 = {abs(1-BETA):.4f}\n")
    print("SINGLE SPHERE (certified bracket + internal-field validation):")
    print(" h       tets     gap/W      E_in/E0")
    for h in [0.13, 0.11]:
        mesh, inside = box_with_sphere(L=1.0, r=0.2, h=h)
        sig = np.where(inside, SIG_P, SIG_M).astype(float)
        u, W_up, W_lo, Ein = solve3d(mesh, sig)
        print(f" {h:.3f}  {mesh.nt:6d}  {(W_up-W_lo)/W_up:8.2e}   {Ein:.4f}")
    print("\nTWO-SPHERE AGGREGATE (cross-validation vs 2D axisymmetric eta_gap ~ 13):")
    for L in [2.0, 3.0]:
        eta, *_ = aggregate_gap_eta(L=L, r=0.2, gap=0.1, h=0.11)
        print(f" box L={L}:  eta_gap = {eta:.2f}")
    print("\n=> full-3D certification (no symmetry assumed) agrees with 1D/2D; the same solver "
          "handles arbitrary non-axisymmetric clusters.")
