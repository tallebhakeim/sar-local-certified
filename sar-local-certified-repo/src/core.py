"""
core.py -- Quasi-static local-field concentration around a nano-inclusion in tissue.

Physics anchor (all closed-form, quasi-static / Rayleigh regime, particle << wavelength):
a homogeneous sphere of complex permittivity eps_p* embedded in tissue eps_m*, exposed to a
uniform applied field E0.  Clausius-Mossotti factor

        beta = (eps_p* - eps_m*) / (eps_p* + 2 eps_m*)

    internal field (uniform)     : E_in  = 3 eps_m* / (eps_p* + 2 eps_m*) * E0
    field just OUTSIDE at pole   : |E|   = |1 + 2 beta| * E0      (theta = 0)
    field just OUTSIDE at equator: |E|   = |1 -   beta| * E0      (theta = 90 deg)

Local SAR ~ sigma_m |E|^2 / (2 rho), so the *local SAR concentration factor* relative to the
bulk (homogeneous-tissue) SAR that dosimetry actually reports is

        eta_pole    = |1 + 2 beta|^2        (up to 9 for a perfect conductor: beta -> 1)
        eta_equator = |1 -   beta|^2

This turns Panagopoulos et al. (PLoS ONE 2013)'s *qualitative* objection -- "SAR is a bulk
average that hides the true local dose" -- into a hard, bounded number.  The magnetic (mu-contrast)
version is identical with eps -> mu and drives the magneto-mechanical force ~ grad(H^2).

Everything here is exact; core.py is the reference the certified bracket (certified_bracket.py)
must contain.
"""
import numpy as np

EPS0 = 8.8541878128e-12   # F/m
MU0  = 4.0e-7 * np.pi     # H/m


def eps_complex(eps_r, sigma, f):
    """Complex permittivity eps* = eps0 eps_r - i sigma/omega  (engineering, exp(+i w t))."""
    omega = 2.0 * np.pi * f
    return EPS0 * eps_r - 1j * sigma / omega


def cm_factor(eps_p, eps_m):
    """Clausius-Mossotti (dipolar polarizability) factor beta."""
    return (eps_p - eps_m) / (eps_p + 2.0 * eps_m)


def enhancement(beta):
    """Return (eta_pole, eta_equator, |E_in/E0|^2) intensity ratios from beta."""
    eta_pole = np.abs(1.0 + 2.0 * beta) ** 2
    eta_eq = np.abs(1.0 - beta) ** 2
    # |E_in/E0|^2 uses E_in = 3 eps_m/(eps_p+2eps_m) E0 = (1 - beta*... ) ; express via beta:
    # 3 eps_m/(eps_p+2eps_m) = 1 - beta  (identity), so internal intensity ratio:
    internal = np.abs(1.0 - beta) ** 2
    return eta_pole, eta_eq, internal


def sar_concentration(eps_p, eps_m):
    """Local SAR concentration factors (medium side, pole & equator) = field intensity ratios."""
    beta = cm_factor(eps_p, eps_m)
    eta_pole, eta_eq, _ = enhancement(beta)
    return dict(beta=beta, eta_pole=float(eta_pole), eta_equator=float(eta_eq))


def field_map(beta, a, r, theta):
    """Exact quasi-static |E|/E0 outside (r>=a) and inside (r<a) the sphere.

    r, theta broadcastable arrays; returns |E|/E0.
    """
    r = np.asarray(r, float)
    theta = np.asarray(theta, float)
    ct, st = np.cos(theta), np.sin(theta)
    out = r >= a
    # outside: E_r = E0 cos(1 + 2 beta a^3/r^3), E_th = -E0 sin(1 - beta a^3/r^3)
    ar3 = np.where(out, (a / np.maximum(r, 1e-30)) ** 3, 0.0)
    Er = ct * (1.0 + 2.0 * beta * ar3)
    Eth = -st * (1.0 - beta * ar3)
    Eout = np.sqrt(np.abs(Er) ** 2 + np.abs(Eth) ** 2)
    # inside: uniform |E_in/E0| = |1 - beta|
    Ein = np.full_like(Eout, np.abs(1.0 - beta))
    return np.where(out, Eout, Ein)


# ---- representative bio / nanoparticle cases -------------------------------------------------

TISSUE_MUSCLE = dict(eps_r=1.0e5, sigma=0.5)   # ~muscle at ~100 kHz (high-permittivity, lossy)

CASES = [
    # name,                eps_r_p,   sigma_p [S/m]
    ("Gold nanoparticle",   1.0,      4.5e7),   # metal -> perfect conductor at these f
    ("Magnetite Fe3O4",     20.0,     2.5e4),   # semiconducting oxide, still highly conductive
    ("Generic conductor",   1.0,      1.0e6),
    ("High-eps dielectric", 80.0,     1e-3),    # e.g. dense dielectric bead
    ("Lipid / void-like",   2.5,      1e-6),    # low-eps inclusion (membrane/lipid, gas bubble)
]


def demo_table(f=1.0e5, tissue=TISSUE_MUSCLE):
    eps_m = eps_complex(tissue["eps_r"], tissue["sigma"], f)
    rows = []
    for name, epr, sig in CASES:
        eps_p = eps_complex(epr, sig, f)
        d = sar_concentration(eps_p, eps_m)
        rows.append((name, d["eta_pole"], d["eta_equator"]))
    return eps_m, rows


if __name__ == "__main__":
    f = 1.0e5
    eps_m, rows = demo_table(f)
    print(f"Local SAR concentration factor eta = |E_local/E0|^2   (tissue=muscle, f={f:.0e} Hz)")
    print(f"  eps_m* = {eps_m:.3e}")
    print(f"  {'inclusion':22s} {'eta_pole':>10s} {'eta_equator':>12s}")
    for name, ep, eq in rows:
        print(f"  {name:22s} {ep:10.3f} {eq:12.3f}")
    print()
    print("  Perfect-conductor limit  eta_pole -> 9  (field x3 at the poles).")
    print("  This factor is entirely averaged out by the regulated bulk SAR.")

    # magneto-mechanical (mu-contrast) sanity: highly permeable sphere -> H_pole = 3 H0 (beta->1)
    beta_mu = cm_factor(1e6, 1.0)   # mu_p/mu_m -> inf
    print(f"\n  Magneto-mechanical: mu_p>>mu_m -> beta={beta_mu.real:.4f}, "
          f"H_pole/H0={np.sqrt(enhancement(beta_mu)[0]):.3f} (force ~ grad H^2).")
