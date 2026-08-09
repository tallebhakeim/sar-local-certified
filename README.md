# Guaranteed bounds on the local dose around magnetic nanoparticle clusters

Computational scripts supporting the article:

> H. Talleb, *How much does averaged dose underestimate the local dose? Guaranteed bounds on the
> specific absorption rate and temperature rise around magnetic nanoparticle clusters*,
> submitted to *International Journal of Hyperthermia*.

Every number, table and figure in the article is produced by the scripts in this repository. Nothing
is entered by hand. This README maps each reported value to the script that computes it, so that any
claim in the paper can be checked by running one file.

## What the code does

Two quantities are enclosed between guaranteed bounds, not estimated:

1. the **local specific absorption rate** around a conductive or magnetic inclusion in tissue, in the
   quasi-static regime;
2. the **local temperature rise** produced by a heated cluster, through the steady Pennes bioheat
   equation.

Both enclosures come from the Prager-Synge complementary variational principle applied on dual
meshes, with hypercircle localization and goal-oriented (dual-weighted-residual) refinement. Neither
interval contains a Friedrichs or Poincaré constant: the bounds are fully computable, valid already
on a coarse mesh, and tighten monotonically under refinement.

## Requirements

Python 3.9 or later, with `numpy`, `scipy` and `matplotlib`:

```sh
pip install -r requirements.txt
```

Every script is standalone and runs in seconds to about half a minute, **with one exception**:
`src/fem3d_certified.py` requires an external discrete-geometric-method core (`dgm`, providing
`mesh3d`, `primal3d`, `mixed3d` and a gmsh interface) that is not distributed here. It reproduces
section 5.4 and figure 11 of the article. The other twelve scripts have no dependency beyond
numpy/scipy/matplotlib.

## Where each result comes from

| Reported in the article | Script | Key output |
|---|---|---|
| Table 1, η values for canonical inclusions | `src/core.py` | Clausius-Mossotti factors, η = 9 at a conductive pole, 2.25 at the equator of a low-contrast bead |
| §5.1, sphere energy bracket (3e-5 → 2e-7) | `src/certified_bracket.py` | two-sided bracket on dissipated power, contains the exact value at every mesh |
| §4.2, hypercircle localization to a voxel | `src/local_bound.py` | local certified interval, radius ρ falling 0.58 → 0.009 |
| §5.1 and figure 6, certified pole concentration | `src/pole_bound.py` | guaranteed lower bound 7.12 against point value 8.65 |
| Table 1 and figure 7, needle tip η = 234.8 | `src/spheroid.py` | prolate spheroid separated exactly; depolarization factor recovered to 5.5e-16 |
| §4.3 and figure 8, arbitrary shape | `src/fem2d.py` | axisymmetric FEM, equilibrated flux from a Stokes stream function; cross-validation with the 1D analytic value to 0.1 % |
| Table 1 and figure 8, aggregate gap η = 13.1 | `src/cluster.py` | two coaxial spheres, no closed form |
| §5.2 and figure 9a, goal-oriented tightening | `src/goal_oriented.py` | half-width ρ_u ρ_z about twentyfold below the global radius |
| §5.2 and figure 9b, exact quadratic SAR | `src/sar_certified_2d.py` | Prudhomme-Oden decomposition, certified η ≥ 2.06 on a coarse mesh |
| §5.3 and figure 10, input uncertainty | `src/uq_intervals.py` | worst-case interval over a conductivity box, exact at the corners |
| §5.4 and figure 11, full 3D | `src/fem3d_certified.py` | body-fitted tetrahedra, RT0 equilibrated flux (**needs the external `dgm` core**) |
| Table 2 and figure 12a, thermal scale analysis | `src/thermal_scale.py` | closed-form Pennes solution; crossover radii 279 µm (φ = 0.5 %) and 88 µm (φ = 5 %) |
| Table 3 and figure 12b, certified ΔT | `src/pennes_certified.py` | energy bracket 0.36 → 0.0056, local interval **[3.0932, 3.1431] K** |
| Figure 12 | `figures/make_figure_thermal.py` | regenerates the figure from the two scripts above |

## Reproducing the headline numbers

```sh
python3 src/thermal_scale.py       # isolated particle 1.3e-9 K; crossover radii
python3 src/pennes_certified.py    # certified temperature interval, 4/4 meshes PASS
python3 src/spheroid.py            # needle tip concentration 234.8
python3 src/cluster.py             # aggregate gap concentration
python3 figures/make_figure_thermal.py
```

Scripts that carry a self-check print `PASS` or `FAIL` per test and return a non-zero exit code on
failure, so the suite can be run unattended.

## Two things worth knowing before reading the code

**The certified quantity is a local average, not a point value.** In three dimensions a
finite-energy field is not continuous, so a pointwise temperature is not bounded in the energy norm.
The quantity of interest is therefore the average over a small ball, which is also what dosimetry
already does when it averages over a mass.

**Quadrature order matters for the certificate, not just for accuracy.** In `pennes_certified.py`
the equilibrated flux is a rational function, so a short Gauss rule under-integrates the
complementary energy and produces an upper bound that is *wrong* rather than merely loose. The
assembly uses 5 points; the certificate uses 24. This is deliberate and is commented in the source.

## Terms of use

These scripts accompany a submitted manuscript and are provided for verification and review. For any
other use, please contact the author.

## Contact

Hakeim Talleb, Sorbonne Université, CNRS, Laboratoire de Génie Électrique et Électronique de Paris
(GeePs), Paris, France. hakeim.talleb@sorbonne-universite.fr
