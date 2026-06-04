# Layla Sphere method

Python reference implementation accompanying the paper

> Bousabaa, A. & Sirolli, R. (2026).
> *The Layla Sphere method: a phase-informed landing algorithm for
> balanced random sampling.* Submitted to *Biometrika*.

The Layla Sphere method is a drop-in replacement for the landing phase of
the Cube method of Deville & Tillé (2004). The flight phase is left
unchanged; the landing phase replaces the greedy "round the closest value
first" rule with a balance-minimising rule informed by phase information
accumulated during the flight. In a simulation study over 4,340 paired
comparisons, the new landing reduces the maximum balancing error by 52.8%
on average, with reductions reaching 68.7% when the number of balancing
variables is large.

## Repository layout

```
.
├── step01_cube_baseline/         Baseline Cube simulation
│   ├── cube_baseline_simulation.py
│   ├── cube_baseline_meta.json   Run metadata
│   └── cube_baseline_raw.csv     Raw output
├── step02_sphere_method/         Layla Sphere implementation + Cube vs Sphere
│   ├── sphere_method.py
│   ├── sphere_vs_cube_meta.json
│   └── sphere_vs_cube_raw.csv
├── step03_pi_ij/                 Second-order inclusion probability study
│   ├── pi_ij_estimation.py
│   ├── sphere_vs_cube_standard_meta.json
│   └── sphere_vs_cube_standard.csv
└── outputs/                      Pre-computed pairwise pi_ij tables
    ├── pi_ij_pairs_*.csv
    └── figures/                  Diagnostic PDF figures
```

> **Note.** The directories `step01_cube_baseline`, `step02_sphere_method`
> and `step03_pi_ij` correspond to the three execution stages of the
> simulation study reported in Section 4 of the paper. Each is
> self-contained and produces both raw output (CSV) and a metadata file
> (JSON) recording the random seed, configuration grid and version
> information.

## Requirements

- Python ≥ 3.9
- `numpy`, `pandas`, `matplotlib`

Install:

```bash
pip install -r requirements.txt
```

## Reproducing the published results

All simulations are deterministic given the random seed. The paper reports
results obtained with **seed 2026** on the configuration grid

| Factor | Levels |
|---|---|
| Population size *N* | 50, 100, 200 |
| Balancing variables *p* | 2, 5, 10 |
| Sampling fraction *f* | 0.2, 0.5, 0.8 |
| Population type | mixed, skewed, correlated |
| Phase weight *α* | 0.0, 0.1, 0.2, 0.4 |

yielding 324 cells × 20 replications = 6,480 individual runs (4,340 valid
paired comparisons after removing degenerate configurations).

Run each step in order:

```bash
# Step 1: Cube baseline (~3,240 Cube draws)
python step01_cube_baseline/cube_baseline_simulation.py

# Step 2: Layla Sphere vs. Cube (paired comparisons)
python step02_sphere_method/sphere_method.py

# Step 3: Empirical pi_ij estimation (Section 5.4 of the paper)
python step03_pi_ij/pi_ij_estimation.py
```

Optional flags (where supported):

- `--test` : small grid (5 configurations) for quick verification
- `--full` : extended grid

The aggregated headline numbers (52.8 %, 68.7 %, 58.1 %, 1.2 %) reported
in Tables 2–5 of the paper are computed from the raw CSV outputs.

## Method summary

For unit *i* with inclusion probability π_i, the Layla Sphere method
represents inclusion as a complex amplitude

    ψ_i = √π_i · exp(i θ_i),       |ψ_i|² = π_i,

so that the amplitude vector lies on the unit sphere
𝒮_{n,N} ⊂ ℂ^N defined by Σ |ψ_i|² = n. The phase θ_i is accumulated
during the flight by

    Δθ_i = λ · u_i / √(max(π_i, ε)),

and is used at landing through a balance-minimising score that combines
the residual balancing error with a phase-coherence term weighted by
α ≥ 0. The flight phase, the stochastic rounding rule, and first-order
inclusion probabilities are identical to those of the Cube method:
the Horvitz–Thompson estimator remains unbiased under the new landing
for any α and any selection order (proved in the Supplementary Material
of the paper).

## Citation

If you use this code, please cite:

```bibtex
@article{BousabaaSirolli2026,
  author  = {Bousabaa, A. and Sirolli, R.},
  title   = {The {L}ayla {S}phere method: a phase-informed landing
             algorithm for balanced random sampling},
  journal = {Biometrika},
  year    = {2026},
  note    = {Submitted}
}
```

A `CITATION.cff` file is provided for GitHub's automatic citation feature.

## References

- Deville, J.-C. & Tillé, Y. (2004). Efficient balanced sampling: the
  cube method. *Biometrika*, **91**, 893–912.

The first production implementation of the Cube method (SAS) was written
in 1999 at ENSAI by A. Bousabaa, J. Lieber and R. Sirolli at the request
of INSEE, from the working paper that later became Deville & Tillé (2004).

## License

MIT — see [LICENSE](LICENSE).
