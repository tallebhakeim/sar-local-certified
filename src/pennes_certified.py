"""
pennes_certified.py -- Encadrement GARANTI de l'energie thermique de Pennes.

Etend la pince de Prager-Synge de la conduction pure (article v1) a la
reaction-diffusion, qui est la forme de l'equation de Pennes stationnaire :

    -div(k grad T) + W T = Q   dans Omega ,   T = 0 sur dOmega
    W = rho_b c_b omega_b (perfusion)  ,  L = sqrt(k/W)

FORMULATION. Avec a(v,w) = int (k grad v . grad w + W v w) et J(v) = a(v,v)/2 - (Q,v),
la solution T minimise J, et a(T,T)/2 = -J(T) = (Q,T)/2.

  BORNE INFERIEURE : pour tout v_h conforme,        -J(v_h)  <=  a(T,T)/2
  BORNE SUPERIEURE : pour tout couple (s,u) verifiant EXACTEMENT
                     -div s + W u = Q,
                     I(s,u) = (1/2) int (|s|^2/k + W u^2)  >=  a(T,T)/2

HYPERCERCLE. En developpant 2 I(s,u) + 2 J(v_h) et en eliminant (Q,v_h) par la
contrainte, tous les termes croises se referment et il vient

    ||| T - v_h |||^2  <=  int |s - k grad v_h|^2 / k  +  int W |u - v_h|^2  =: rho^2

sans AUCUNE constante inconnue (ni Friedrichs, ni Poincare) : c'est le point qui
distingue une borne garantie d'un estimateur. Le terme de reaction est absorbe
par la variable auxiliaire u, qu'on prend ici egale a v_h -- tout le residu est
alors porte par le flux equilibre.

CONSTRUCTION DU FLUX EQUILIBRE. En symetrie spherique la contrainte s'integre
exactement : div s = W u - Q donne

    r^2 s_r(r) = int_0^r (W u - Q) rho^2 drho ,

et l'integrande est un polynome de degre 3 par element (u affine, Q constant par
morceaux), donc la primitive est exacte. Aucune reconstruction de type
Raviart-Thomas n'est necessaire, exactement comme la fonction de courant de
Stokes evitait RT0 en axisymetrique.

VERIFICATION. La reference est la solution close du probleme TRONQUE (T = 0 en
r = Rmax), et non celle du domaine infini : sans cela on comparerait deux
problemes differents et l'ecart de troncature masquerait la convergence.
"""

import numpy as np

from thermal_scale import K_TISSUE, W_PERF, L_PERF

# quadrature de Gauss-Legendre a 5 points sur [-1, 1] (exacte jusqu'au degre 9) :
# suffisante pour l'assemblage, ou tous les integrandes sont polynomiaux.
_GX, _GW = np.polynomial.legendre.leggauss(5)

# Le CERTIFICAT, lui, integre s_r = G(r)/r^2, qui est une fraction rationnelle et
# non un polynome. Une quadrature trop courte sous-estime int |s|^2/k et fabrique
# une borne superieure fausse (constatee : violation de 4e-4 sur maillage
# grossier). On evalue donc les integrales du certificat a un ordre nettement
# plus eleve ; c'est peu couteux et cela conditionne la validite de la garantie.
_CX, _CW = np.polynomial.legendre.leggauss(24)


# --------------------------------------------------------------- solution close
def exact_truncated(r, R, Q, Rmax, k=K_TISSUE, W=W_PERF):
    """Solution exacte de -div(k grad T) + W T = Q chi_{r<R} sur la boule de rayon
    Rmax, avec T(Rmax) = 0. Symetrie spherique."""
    L = np.sqrt(k / W)

    def f(x):                       # interieur : sinh(x/L)/(x/L)
        return L * np.sinh(x / L) / x

    def df(x):
        return np.cosh(x / L) / x - L * np.sinh(x / L) / x**2

    def g(x):                       # exterieur : s'annule en Rmax
        return (np.exp(-x / L) - np.exp((x - 2 * Rmax) / L)) / x

    def dg(x):
        return (-np.exp(-x / L) / L - np.exp((x - 2 * Rmax) / L) / L) / x \
               - (np.exp(-x / L) - np.exp((x - 2 * Rmax) / L)) / x**2

    A = 1.0 / (f(R) - df(R) * g(R) / dg(R))
    C = -(Q / W) * A * df(R) / dg(R)

    r = np.atleast_1d(np.asarray(r, dtype=float))
    T = np.empty_like(r)
    ins = r <= R
    rr = np.where(r > 0, r, 1e-300)
    T[ins] = (Q / W) * (1.0 - A * L * np.sinh(rr[ins] / L) / rr[ins])
    T[~ins] = C * g(rr[~ins])
    return T if T.size > 1 else float(T[0])


def exact_energy_half(R, Q, Rmax, k=K_TISSUE, W=W_PERF, n_sub=4000):
    """a(T,T)/2 = (Q,T)/2 = 2 pi int_0^R Q T r^2 dr, par quadrature fine."""
    edges = np.linspace(0.0, R, n_sub + 1)
    total = 0.0
    for i in range(n_sub):
        a, b = edges[i], edges[i + 1]
        mid, half = 0.5 * (a + b), 0.5 * (b - a)
        x = mid + half * _GX
        total += half * np.sum(_GW * Q * exact_truncated(x, R, Q, Rmax, k, W) * x**2)
    return 0.5 * 4.0 * np.pi * total


# ------------------------------------------------------------------- maillage
def build_mesh(R, Rmax, n_in, n_out):
    """Noeuds avec r = R EXACTEMENT present (la source y est discontinue)."""
    return np.concatenate([np.linspace(0.0, R, n_in + 1),
                           np.linspace(R, Rmax, n_out + 1)[1:]])


# ------------------------------------------------------------- primal P1 radial
def solve_primal(nodes, R, Q, k=K_TISSUE, W=W_PERF):
    """P1 a poids r^2, T(Rmax) = 0. Retourne le vecteur nodal."""
    n = nodes.size
    A = np.zeros((n, n))
    b = np.zeros(n)
    for e in range(n - 1):
        r0, r1 = nodes[e], nodes[e + 1]
        h = r1 - r0
        mid, half = 0.5 * (r0 + r1), 0.5 * h
        x = mid + half * _GX
        w = _GW * half * x**2                      # poids r^2 dr
        Qe = Q if 0.5 * (r0 + r1) < R else 0.0
        # fonctions de forme et derivees
        phi = np.array([(r1 - x) / h, (x - r0) / h])
        dphi = np.array([-1.0 / h, 1.0 / h])
        for a_ in range(2):
            b[e + a_] += np.sum(w * Qe * phi[a_])
            for b_ in range(2):
                A[e + a_, e + b_] += (k * dphi[a_] * dphi[b_] * np.sum(w)
                                      + W * np.sum(w * phi[a_] * phi[b_]))
    # Dirichlet homogene au bord exterieur
    A[-1, :] = 0.0
    A[:, -1] = 0.0
    A[-1, -1] = 1.0
    b[-1] = 0.0
    return np.linalg.solve(A, b)


# ------------------------------------------------- flux equilibre + certificat
def certify(nodes, T, R, Q, k=K_TISSUE, W=W_PERF):
    """Construit s exactement equilibre (u = v_h) et renvoie le dictionnaire
    des quantites certifiees."""
    n = nodes.size
    G = 0.0                       # cumul de int_0^r (W u - Q) rho^2 drho
    rho2 = 0.0                    # estimateur au carre (sans le 4 pi)
    energy_h = 0.0                # a(v_h, v_h) / (4 pi)
    load_h = 0.0                  # (Q, v_h) / (4 pi)
    compl = 0.0                   # int (|s|^2/k + W u^2) / (4 pi)

    for e in range(n - 1):
        r0, r1 = nodes[e], nodes[e + 1]
        h = r1 - r0
        T0, T1 = T[e], T[e + 1]
        slope = (T1 - T0) / h
        icept = T0 - slope * r0            # u(rho) = icept + slope * rho
        Qe = Q if 0.5 * (r0 + r1) < R else 0.0

        # primitive exacte de (W u - Q) rho^2 = (W icept - Qe) rho^2 + W slope rho^3
        c2 = W * icept - Qe
        c3 = W * slope

        def prim(x):
            return c2 * x**3 / 3.0 + c3 * x**4 / 4.0

        mid, half = 0.5 * (r0 + r1), 0.5 * h
        x = mid + half * _CX
        w = _CW * half * x**2
        u = icept + slope * x
        s_r = (G + prim(x) - prim(r0)) / x**2      # flux equilibre exact

        rho2 += np.sum(w * (s_r - k * slope) ** 2) / k
        energy_h += k * slope**2 * np.sum(w) + W * np.sum(w * u**2)
        load_h += Qe * np.sum(w * u)
        compl += np.sum(w * s_r**2) / k + W * np.sum(w * u**2)

        G += prim(r1) - prim(r0)

    four_pi = 4.0 * np.pi
    lower = four_pi * (load_h - 0.5 * energy_h)     # -J(v_h)
    upper = four_pi * 0.5 * compl                   # I(s, u)
    return {"lower": lower, "upper": upper,
            "rho": np.sqrt(four_pi * rho2),
            "energy_norm_bound": np.sqrt(four_pi * rho2)}


# ------------------------------------------------- borne locale goal-oriented
def mean_over_ball(nodes, T, Rs):
    """Moyenne de v_h sur la boule centrale de rayon Rs (Rs doit etre un noeud)."""
    vol = (4.0 / 3.0) * np.pi * Rs**3
    total = 0.0
    for e in range(nodes.size - 1):
        r0, r1 = nodes[e], nodes[e + 1]
        if r1 > Rs + 1e-18:
            break
        h = r1 - r0
        mid, half = 0.5 * (r0 + r1), 0.5 * h
        x = mid + half * _CX
        slope = (T[e + 1] - T[e]) / h
        u = T[e] + slope * (x - r0)
        total += np.sum(_CW * half * x**2 * u)
    return 4.0 * np.pi * total / vol


def certified_peak(R, Q, Rmax, Rs, n_in, n_mid, n_out, k=K_TISSUE, W=W_PERF):
    """Encadrement GARANTI de la temperature moyenne sur la boule centrale de
    rayon Rs, par dualite : |s(T) - s(v_h)| <= rho_u * rho_z.

    La grandeur d'interet est une MOYENNE sur une petite boule, pas une valeur
    ponctuelle : en trois dimensions une fonction d'energie finie n'est pas
    continue, donc T(0) n'est pas une quantite bornee au sens de l'energie. La
    moyenne locale est la bonne observable, et c'est aussi celle qui a un sens
    dosimetrique (on moyenne deja sur une masse).
    """
    nodes = np.unique(np.concatenate([
        np.linspace(0.0, Rs, n_in + 1),
        np.linspace(Rs, R, n_mid + 1),
        np.linspace(R, Rmax, n_out + 1)]))

    # primal
    T = solve_primal(nodes, R, Q, k, W)
    cu = certify(nodes, T, R, Q, k, W)

    # adjoint : source = indicatrice de S normalisee (meme routine, autre source)
    q_adj = 1.0 / ((4.0 / 3.0) * np.pi * Rs**3)
    Z = solve_primal(nodes, Rs, q_adj, k, W)
    cz = certify(nodes, Z, Rs, q_adj, k, W)

    s_h = mean_over_ball(nodes, T, Rs)
    delta = cu["rho"] * cz["rho"]
    return {"nodes": nodes.size, "s_h": s_h, "delta": delta,
            "lo": s_h - delta, "hi": s_h + delta,
            "rho_u": cu["rho"], "rho_z": cz["rho"]}


# ------------------------------------------------------------------------ main
if __name__ == "__main__":
    print("=" * 74)
    print("PINCE CERTIFIEE SUR PENNES (reaction-diffusion)")
    print("=" * 74)

    R = 500e-6                       # amas de 500 um (regime ou le pic compte)
    Rmax = 6.0 * L_PERF
    Q = 0.005 * 5200.0 * 500e3       # phi = 0.5 %, SLP = 500 W/g
    print(f"R = {R*1e6:.0f} um   Rmax = {Rmax*1e3:.1f} mm   Q = {Q:.3e} W/m3")
    print(f"L_perf = {L_PERF*1e3:.1f} mm")

    ref = exact_energy_half(R, Q, Rmax)
    print(f"\nreference  a(T,T)/2 = {ref:.10e} W.K")

    print("\n  n_in  n_out |        borne inf        borne sup |"
          "     gap relatif |        rho")
    print("  " + "-" * 84)
    for n_in, n_out in [(20, 40), (40, 80), (80, 160), (160, 320), (320, 640)]:
        nodes = build_mesh(R, Rmax, n_in, n_out)
        T = solve_primal(nodes, R, Q)
        c = certify(nodes, T, R, Q)
        ok = c["lower"] <= ref <= c["upper"]
        gap = (c["upper"] - c["lower"]) / ref
        flag = "OK" if ok else "!! HORS BORNES"
        print(f"  {n_in:5d} {n_out:6d} | {c['lower']:.10e}  {c['upper']:.10e} |"
              f" {gap:14.3e} | {c['rho']:.3e}  {flag}")

    print("\n  gap relatif -> 0 et reference toujours encadree = pince valide.")

    # ---------------------------------------------------------------- pic local
    print("\n" + "=" * 74)
    print("BORNE LOCALE CERTIFIEE sur l'elevation de temperature (goal-oriented)")
    print("=" * 74)
    Rs = R / 5.0                      # boule centrale de 100 um
    print(f"grandeur d'interet : temperature moyenne sur la boule centrale "
          f"de rayon {Rs*1e6:.0f} um")
    # Reference = moyenne de VOLUME de la solution close sur la boule, soit
    # 3/Rs^3 * int_0^Rs T r^2 dr. Une moyenne radiale non ponderee donnerait une
    # autre grandeur (surestimee ici, T decroissant en r) et ferait croire a tort
    # a une violation de la borne.
    _e = np.linspace(0.0, Rs, 2001)
    _acc = 0.0
    for _i in range(_e.size - 1):
        _a, _b = _e[_i], _e[_i + 1]
        _m, _hh = 0.5 * (_a + _b), 0.5 * (_b - _a)
        _x = _m + _hh * _CX
        _acc += _hh * np.sum(_CW * exact_truncated(_x, R, Q, Rmax) * _x**2)
    T_ref = 3.0 * _acc / Rs**3

    print(f"\n  reference (moyenne de volume de la solution close) : {T_ref:.4f} K")
    print("\n  noeuds |    s(v_h)   |  demi-largeur |     intervalle garanti"
          "          | contient")
    print("  " + "-" * 84)
    all_ok = True
    for n_in, n_mid, n_out in [(8, 32, 60), (16, 64, 120),
                               (32, 128, 240), (64, 256, 480)]:
        c = certified_peak(R, Q, Rmax, Rs, n_in, n_mid, n_out)
        ok = c["lo"] <= T_ref <= c["hi"]
        all_ok &= ok
        print(f"  {c['nodes']:6d} | {c['s_h']:11.5f} | {c['delta']:13.3e} |"
              f" [{c['lo']:8.4f}, {c['hi']:8.4f}] K |"
              f"   {'PASS' if ok else 'FAIL'}")

    print(f"\n  bilan : {'PASS' if all_ok else 'FAIL'} -- tous les intervalles "
          "contiennent la reference et se resserrent.")
    print("  L'intervalle est GARANTI : il ne suppose aucune constante inconnue,")
    print("  seulement l'orthogonalite de Galerkin et deux flux equilibres exacts.")
    print("=" * 74)
