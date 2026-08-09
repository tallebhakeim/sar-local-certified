"""
uq_intervals.py -- interval (worst-case) uncertainty quantification on the local SAR concentration.

The certified bracket already controls the NUMERICAL (discretization) uncertainty. Here we add the
PARAMETRIC uncertainty: tissue and particle conductivities are known only within intervals. We
produce a GUARANTEED eta-interval over the whole parameter box, so the certificate is doubly
guaranteed (over the computation AND over the inputs).

Key fact making it rigorous and cheap: for the quasi-static conduction contrast, the pole
concentration eta_pole = (1 + 2 beta)^2 with beta = (s-1)/(s+2), s = sigma_p/sigma_m, is MONOTONE
increasing in s (dbeta/ds = 3/(s+2)^2 > 0). Hence over sigma_p in [p_lo,p_hi], sigma_m in [m_lo,m_hi]
the extremes are attained at the corners:
    s_min = p_lo/m_hi  -> eta_min ,    s_max = p_hi/m_lo  -> eta_max .
No interior sampling is needed; the interval is exact.

For an arbitrary FEM shape the same monotonicity (by the comparison principle) lets the parameter
box combine with the per-corner certified bracket: eta in [min_corner eta_lower, max_corner eta_upper].
"""
import numpy as np


def eta_pole(s):
    beta = (s - 1.0) / (s + 2.0)
    return (1.0 + 2.0 * beta) ** 2


def eta_interval(m_lo, m_hi, p_lo, p_hi):
    """Guaranteed [eta_min, eta_max] over sigma_m in [m_lo,m_hi], sigma_p in [p_lo,p_hi]."""
    s_min = p_lo / m_hi
    s_max = p_hi / m_lo
    return eta_pole(s_min), eta_pole(s_max)


# nominal cases (tissue-like sigma_m ~ 0.5 S/m)
SM_NOM = 0.5
CASES = {
    "metallic particle (sigma_p ~ 1e6)": 1.0e6,
    "moderate conductor (sigma_p = 5)": 5.0,
}


def band_vs_uncertainty(sp_nom, sm_nom=SM_NOM, plevels=None):
    if plevels is None:
        plevels = np.linspace(0.0, 0.5, 26)
    lo, hi = [], []
    for p in plevels:
        a, b = eta_interval(sm_nom * (1 - p), sm_nom * (1 + p),
                            sp_nom * (1 - p), sp_nom * (1 + p))
        lo.append(a); hi.append(b)
    return plevels, np.array(lo), np.array(hi)


if __name__ == "__main__":
    print("=== Interval (worst-case) UQ on the pole SAR concentration eta ===")
    print(f"tissue sigma_m nominal = {SM_NOM} S/m ; eta_pole is monotone in sigma_p/sigma_m\n")
    for name, sp in CASES.items():
        print(f"--- {name}")
        for p in (0.10, 0.30, 0.50):
            a, b = eta_interval(SM_NOM * (1 - p), SM_NOM * (1 + p), sp * (1 - p), sp * (1 + p))
            print(f"    +-{int(p*100):2d}% uncertainty:  eta guaranteed in [{a:.3f}, {b:.3f}]"
                  f"   (lower bound {'>' if a > 1 else '<='} 1)")
        print()
    print("Takeaway: even at +-50% tissue/particle conductivity uncertainty, the guaranteed lower "
          "bound on eta stays well above 1, so the certified local concentration does not hinge on "
          "precise tissue data.")
