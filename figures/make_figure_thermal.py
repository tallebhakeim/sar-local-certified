"""Figure du volet thermique de l'article Int. J. Hyperthermia.

(a) elevation de temperature au centre d'un amas charge en nanoparticules, en
    fonction du rayon de l'amas : la loi en R^2 et le rayon de croisement.
(b) encadrement GARANTI de l'elevation locale, qui se resserre autour de la
    reference sous raffinement.

Les donnees viennent directement des scripts du PoC ; rien n'est saisi a la main.
Lancer :  python3 make_figure_thermal.py
"""

import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import os
POC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
sys.path.insert(0, POC)

from thermal_scale import (peak_temperature, crossover_radius,   # noqa: E402
                           RHO_NP, SLP_SI, L_PERF)
from pennes_certified import certified_peak, exact_truncated, _CX, _CW  # noqa: E402

plt.rcParams.update({"font.size": 9, "axes.labelsize": 9,
                     "axes.titlesize": 9.5, "legend.fontsize": 8})

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.0, 4.2))

# ---------------------------------------------------------------- panneau (a)
radii = np.logspace(-8, -2, 220)              # 10 nm -> 10 mm
for phi, style in ((0.005, "-"), (0.05, "--")):
    Q = phi * RHO_NP * SLP_SI
    dT = np.array([peak_temperature(R, Q) for R in radii])
    ax1.loglog(radii * 1e6, dT, style, lw=1.8,
               label=f"$\\varphi$ = {phi*100:g} %")

ax1.axhline(1.0, color="0.45", lw=1.0, ls=":")
ax1.text(1.3e-2, 1.8, "1 K", color="0.35", fontsize=8)

# Les deux rayons de croisement sont proches : on les annote de part et d'autre
# de la ligne 1 K pour qu'ils ne se recouvrent pas.
for phi, col, fx, fy in ((0.05, "tab:orange", 0.06, 2.5e2),
                         (0.005, "tab:blue", 2.6, 1.5e-2)):
    Rc = crossover_radius(phi, 1.0)
    ax1.plot([Rc * 1e6], [1.0], "o", color=col, ms=5.5, zorder=5)
    ax1.annotate(f"{Rc*1e6:.0f} µm", xy=(Rc * 1e6, 1.0),
                 xytext=(Rc * 1e6 * fx, fy), color=col, fontsize=8,
                 arrowprops=dict(arrowstyle="->", color=col, lw=0.9))

ax1.axvspan(1e-2, 1.0, color="0.88", alpha=0.55, zorder=0)
ax1.text(1.3e-2, 2e-4, "isolated\nnanoparticle", fontsize=8, color="0.35")
ax1.axvline(1000.0, color="0.5", lw=1.0, ls="-.")
ax1.text(1250.0, 2e-7, "1 mm cluster\n(Rabin 2002)", fontsize=8, color="0.35")

ax1.set_xlabel("cluster radius  $R$  (µm)")
ax1.set_ylabel("peak temperature rise at centre  $\\Delta T$  (K)")
ax1.set_title("(a)  Diffusion erases the peak below ~100 µm", loc="left")
ax1.set_ylim(1e-10, 1e5)
ax1.grid(True, which="major", alpha=0.3)
ax1.legend(loc="upper left", frameon=False)

# ---------------------------------------------------------------- panneau (b)
R = 500e-6
Rmax = 6.0 * L_PERF
Q = 0.005 * RHO_NP * SLP_SI
Rs = R / 5.0

_e = np.linspace(0.0, Rs, 2001)
_acc = 0.0
for _i in range(_e.size - 1):
    _a, _b = _e[_i], _e[_i + 1]
    _m, _h = 0.5 * (_a + _b), 0.5 * (_b - _a)
    _x = _m + _h * _CX
    _acc += _h * np.sum(_CW * exact_truncated(_x, R, Q, Rmax) * _x**2)
T_ref = 3.0 * _acc / Rs**3

grids = [(8, 32, 60), (16, 64, 120), (32, 128, 240), (64, 256, 480)]
n_nodes, lo, hi = [], [], []
for g in grids:
    c = certified_peak(R, Q, Rmax, Rs, *g)
    n_nodes.append(c["nodes"])
    lo.append(c["lo"])
    hi.append(c["hi"])
n_nodes, lo, hi = np.array(n_nodes), np.array(lo), np.array(hi)

ax2.fill_between(n_nodes, lo, hi, color="tab:green", alpha=0.22,
                 label="guaranteed interval")
ax2.plot(n_nodes, lo, "o-", color="tab:green", lw=1.6, ms=5,
         label="certified lower bound")
ax2.plot(n_nodes, hi, "s-", color="tab:red", lw=1.6, ms=5,
         label="certified upper bound")
ax2.axhline(T_ref, color="0.25", lw=1.2, ls="--",
            label=f"exact reference ({T_ref:.3f} K)")

ax2.set_xscale("log")
ax2.set_xlabel("number of nodes")
ax2.set_ylabel("$\\Delta T$ averaged over the 100 µm ball  (K)")
ax2.set_title("(b)  The enclosure tightens, with no unknown constant",
              loc="left")
ax2.grid(True, alpha=0.3)
ax2.legend(loc="lower right", frameon=False)

fig.tight_layout()
out = "fig_thermal.png"
fig.savefig(out, dpi=200, bbox_inches="tight")
print(f"ecrit {out}")
print(f"  reference = {T_ref:.4f} K")
print(f"  intervalle le plus fin = [{lo[-1]:.4f}, {hi[-1]:.4f}] K")
