"""
thermal_scale.py -- Diagnostic d'echelle : le pic de SAR local survit-il a la
diffusion thermique ?

Question posee par la communaute hyperthermie (Rabin 2002, Keblinski 2006) :
une nanoparticule isolee ne chauffe pas son voisinage, car la diffusion efface
le pic. On calcule ici, en forme close, a partir de quelle taille d'amas la
concentration locale de dose devient thermiquement visible.

Modele : Pennes stationnaire dans le milieu autour d'une region source spherique
de rayon R dissipant une puissance volumique uniforme Q :

    k Lap(T) - W (T - Ta) + Q = 0 ,   W = rho_b c_b omega_b ,   L = sqrt(k/W)

Solution exacte a symetrie spherique (T mesuree en exces sur Ta), continue en
valeur et en flux en r = R, bornee a l'origine et a l'infini :

    interieur : T(r) = Q/W * [ 1 - A * sinh(r/L)/(r/L) ]
    exterieur : T(r) = Q/W * B * exp(-r/L) / (r/L)

avec A, B fixes par la continuite. Le maximum est en r = 0.

Sortie : (1) verification de la solution close contre une resolution numerique
directe de l'EDO ; (2) le balayage en R qui donne la longueur de croisement.
"""

import numpy as np
from scipy.linalg import solve_banded

# ---------------------------------------------------------------- proprietes
# tissu mou / tumeur, valeurs usuelles de la litterature dosimetrique
K_TISSUE = 0.50        # conductivite thermique          [W/(m.K)]
RHO_B = 1000.0         # masse volumique du sang         [kg/m3]
C_B = 3600.0           # chaleur massique du sang        [J/(kg.K)]
OMEGA_B = 8.0e-4       # debit de perfusion              [1/s]  (~0.8 mL/(g.min))
W_PERF = RHO_B * C_B * OMEGA_B          # [W/(m3.K)]
L_PERF = np.sqrt(K_TISSUE / W_PERF)     # longueur de perfusion [m]

# nanoparticules d'oxyde de fer en hyperthermie magnetique
RHO_NP = 5200.0        # magnetite                        [kg/m3]
SLP = 500.0            # puissance specifique d'absorption [W/g de NP]
SLP_SI = SLP * 1e3     # [W/kg]


def temperature_profile(r, R, Q, k=K_TISSUE, W=W_PERF):
    """Exces de temperature exact pour une source spherique uniforme de rayon R.

    r peut etre scalaire ou tableau. Retourne T(r) - Ta en kelvins.
    """
    L = np.sqrt(k / W)
    x = R / L
    # continuite de T et de k dT/dr en r = R
    #   interieur  T = (Q/W) [1 - A sinh(r/L)/(r/L)]
    #   exterieur  T = (Q/W) B exp(-r/L)/(r/L)
    # -> A = exp(-x) (1 + 1/x) / (cosh(x)/x - sinh(x)/x**2) / x  ... resolu ci-dessous
    # On resout le systeme 2x2 explicitement.
    #  f_i(x)  = sinh(x)/x           ; f_i'(x) = (cosh(x) - sinh(x)/x)/x
    #  f_e(x)  = exp(-x)/x           ; f_e'(x) = -exp(-x)(1/x + 1/x**2)
    f_i = np.sinh(x) / x
    df_i = (np.cosh(x) - np.sinh(x) / x) / x
    f_e = np.exp(-x) / x
    df_e = -np.exp(-x) * (1.0 / x + 1.0 / x**2)
    # 1 - A f_i = B f_e        (continuite de T)
    # -A df_i   = B df_e       (continuite du flux)
    # -> A (df_e f_i - df_i f_e) = df_e  ->  A = df_e / (df_e f_i - df_i f_e)
    A = df_e / (df_e * f_i - df_i * f_e)
    B = -A * df_i / df_e

    r = np.atleast_1d(np.asarray(r, dtype=float))
    T = np.empty_like(r)
    inside = r <= R
    # limite r -> 0 : sinh(u)/u -> 1
    u = np.where(r > 0, r / L, 1e-300)
    T[inside] = (Q / W) * (1.0 - A * np.sinh(u[inside]) / u[inside])
    T[~inside] = (Q / W) * B * np.exp(-u[~inside]) / u[~inside]
    return T if T.size > 1 else float(T[0])


def peak_temperature(R, Q, k=K_TISSUE, W=W_PERF):
    """Exces de temperature au centre (r -> 0) de la source spherique."""
    L = np.sqrt(k / W)
    x = R / L
    f_i = np.sinh(x) / x
    df_i = (np.cosh(x) - np.sinh(x) / x) / x
    f_e = np.exp(-x) / x
    df_e = -np.exp(-x) * (1.0 / x + 1.0 / x**2)
    A = df_e / (df_e * f_i - df_i * f_e)
    return (Q / W) * (1.0 - A)          # sinh(u)/u -> 1 quand u -> 0


# ------------------------------------------------------- verification numerique
def _verify_against_ode(R, Q, cells_in_source=400, r_max_factor=6.0):
    """Resout k/r^2 d/dr(r^2 dT/dr) - W T + Q chi_{r<R} = 0 par differences finies
    et compare au profil ferme. Retourne l'ecart relatif max.

    Le pas est choisi de sorte que r = R tombe EXACTEMENT sur un noeud : sinon la
    discontinuite de source est etalee sur une maille et l'erreur est en O(h),
    pas en O(h^2), ce qui masque la verification.
    """
    r_max = max(r_max_factor * R, 6.0 * L_PERF)
    h = R / cells_in_source
    n = int(round(r_max / h)) + 1
    r = np.arange(n) * h
    # substitution u = r T  ->  k u'' - W u + Q r chi = 0 , u(0) = 0 , u(r_max) = 0
    N = n - 2                                   # inconnues interieures u_1..u_N
    ri = r[1:-1]
    main = np.full(N, -2.0 * K_TISSUE / h**2 - W_PERF)
    off = np.full(N - 1, K_TISSUE / h**2)
    # fraction de la cellule duale [r-h/2, r+h/2] situee dans la source : vaut 1,
    # 0, ou 1/2 exactement au noeud r = R. Sans cela l'erreur retombe en O(h).
    frac = np.clip((R - (ri - 0.5 * h)) / h, 0.0, 1.0)
    rhs = -Q * ri * frac
    ab = np.zeros((3, N))
    ab[0, 1:] = off
    ab[1, :] = main
    ab[2, :-1] = off
    u = solve_banded((1, 1), ab, rhs)
    T_num = u / ri
    T_ref = temperature_profile(ri, R, Q)
    return np.max(np.abs(T_num - T_ref)) / np.max(np.abs(T_ref))


# ------------------------------------------------------------------ diagnostic
def cluster_sweep(volume_fraction=0.005):
    """Balayage de la taille d'amas. La source est un amas spherique de rayon R
    charge a la fraction volumique donnee en nanoparticules dissipant SLP."""
    Q = volume_fraction * RHO_NP * SLP_SI     # puissance volumique de l'amas [W/m3]
    radii = np.array([10e-9, 100e-9, 1e-6, 10e-6, 50e-6, 100e-6,
                      500e-6, 1e-3, 5e-3, 10e-3])
    return Q, radii, np.array([peak_temperature(R, Q) for R in radii])


def crossover_radius(volume_fraction=0.005, dT_target=1.0):
    """Rayon d'amas pour lequel le pic atteint dT_target (bissection)."""
    Q = volume_fraction * RHO_NP * SLP_SI
    lo, hi = 1e-9, 1.0
    for _ in range(200):
        mid = np.sqrt(lo * hi)
        if peak_temperature(mid, Q) < dT_target:
            lo = mid
        else:
            hi = mid
    return np.sqrt(lo * hi)


if __name__ == "__main__":
    print("=" * 74)
    print("DIAGNOSTIC D'ECHELLE THERMIQUE -- le pic de dose locale survit-il ?")
    print("=" * 74)
    print(f"k = {K_TISSUE} W/(m.K)   W = {W_PERF:.0f} W/(m3.K)   "
          f"L_perf = {L_PERF*1e3:.1f} mm")
    print(f"NP : rho = {RHO_NP} kg/m3, SLP = {SLP} W/g")

    print("\n[1] Verification de la solution close contre l'EDO discretisee")
    for R in (100e-6, 1e-3, 5e-3):
        err = _verify_against_ode(R, Q=1e5)
        flag = "PASS" if err < 1e-3 else "FAIL"
        print(f"    R = {R*1e3:7.3f} mm   ecart relatif max = {err:.2e}   {flag}")

    print("\n[2] Pic de temperature au centre d'un amas de nanoparticules")
    for phi in (0.005, 0.05):
        Q, radii, dT = cluster_sweep(phi)
        print(f"\n    fraction volumique phi = {phi*100:g} %   "
              f"-> Q_amas = {Q:.3e} W/m3")
        print("      rayon d'amas R        pic dT au centre")
        for R, t in zip(radii, dT):
            unit = f"{R*1e9:8.1f} nm" if R < 1e-6 else (
                   f"{R*1e6:8.1f} um" if R < 1e-3 else f"{R*1e3:8.2f} mm")
            print(f"      {unit}          {t:12.3e} K")
        Rc = crossover_radius(phi, dT_target=1.0)
        print(f"      -> rayon de croisement a dT = 1 K : {Rc*1e6:.1f} um")

    print("\n[3] Nanoparticule ISOLEE (le contre-argument Rabin / Keblinski)")
    a = 10e-9
    Q_np = RHO_NP * SLP_SI          # toute la particule dissipe
    dT_np = peak_temperature(a, Q_np)
    print(f"    a = {a*1e9:.0f} nm, Q = {Q_np:.2e} W/m3  ->  dT = {dT_np:.3e} K")
    print("    -> confirme : le canal thermique est mort pour la NP isolee.")
    print("=" * 74)
