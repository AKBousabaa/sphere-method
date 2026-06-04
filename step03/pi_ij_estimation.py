"""
=============================================================
  ESTIMATION EMPIRIQUE DES π_ij — CUBE vs SPHÈRE DE LAYLA
  python/step03/pi_ij_estimation.py

  Objectif :
    Pour une population FIXÉE, estimer empiriquement les
    probabilités d'inclusion du second ordre π_ij pour le
    Cube (Deville & Tillé 2004) et la Sphère de Layla
    (Bousabaa & Sirolli 2026) par simulation Monte Carlo.

    Contrairement aux scripts step01 et step02 qui régénèrent
    une nouvelle population à chaque réplication, ici la
    population est fixée une fois pour toutes : π_ij est une
    propriété du DESIGN, pas de la population.

  Fondement théorique :
    La formule de Sen–Yates–Grundy (Appendice B3) donne :

      Var(Ŷ_HT) = -Σ_{i<j} (π_ij - π_i π_j)(y_i/π_i - y_j/π_j)²

    La condition B3 (corrélations négatives induites) affirme :
      π_ij^Sphère ≤ π_ij^Cube  pour les paires à grand contraste.

    Ce script teste cette condition empiriquement.

  Questions explorées :
    Q1 : π_ij^Sphère ≤ π_ij^Cube pour quelle fraction des paires ?
    Q2 : L'inégalité est-elle concentrée sur les grands contrastes
         (y_i/π_i - y_j/π_j)² ? (lien direct avec la variance SYG)
    Q3 : Les deux méthodes induisent-elles des covariances
         π_ij - π_i π_j négatives ? La Sphère plus que le Cube ?
    Q4 : La distance auxiliaire Σ_k(A_ki - A_kj)² prédit-elle
         le signe de π_ij^Sphère - π_i π_j ?
    Q5 : Variance SYG empirique — confirme-t-elle le résultat RMSE ?

  Résultats exportés :
    outputs/pi_ij_pairs_<config>.csv   : données par paire (i,j)
    outputs/pi_ij_summary.json         : statistiques agrégées
    outputs/figures/                   : figures (matplotlib)

  Usage :
    python pi_ij_estimation.py              # standard (N=50, R=5000)
    python pi_ij_estimation.py --test       # rapide  (N=30, R=500)
    python pi_ij_estimation.py --full       # exhaustif (multi-config)

  Auteurs : A. Bousabaa & R. Sirolli (2026)
  Seed principal : 2026
=============================================================
"""

import numpy as np
import pandas as pd
import json
import time
import os
import sys
import warnings

warnings.filterwarnings("ignore")

# ── Matplotlib optionnel ────────────────────────────────────
try:
    import matplotlib
    matplotlib.use("Agg")          # pas d'affichage interactif
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("  [INFO] matplotlib absent — figures ignorées.")

EPS     = 1e-10
EPS_INT = 1e-8


# ============================================================
# DOSSIERS DE SORTIE
# ============================================================

def _output_dir():
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "..", "outputs")
    os.makedirs(base, exist_ok=True)
    fig = os.path.join(base, "figures")
    os.makedirs(fig, exist_ok=True)
    return base, fig

OUT_DIR, FIG_DIR = _output_dir()


# ============================================================
# UTILITAIRES COMMUNS  (auto-contenus, cohérents avec step02)
# ============================================================

def _null_vec(A: np.ndarray):
    """Premier vecteur du noyau de A via SVD."""
    if A.size == 0 or A.shape[1] == 0:
        return None
    try:
        _, s, Vt = np.linalg.svd(A, full_matrices=True)
        maxs = float(s[0]) if len(s) > 0 else 1e-12
        rank = int(np.sum(s > 1e-9 * max(maxs, 1e-12)))
        if rank >= A.shape[1]:
            return None
        u = Vt[rank].copy()
        norm = np.linalg.norm(u)
        return u / norm if norm > EPS else None
    except np.linalg.LinAlgError:
        return None


def flight_phase(pi: np.ndarray, A: np.ndarray,
                 rng: np.random.Generator,
                 track_phase: bool = False) -> tuple:
    """Phase de vol commune Cube / Sphère de Layla."""
    pi    = pi.copy()
    N     = len(pi)
    theta = np.zeros(N) if track_phase else None

    for _ in range(500_000):
        active = np.where((pi > EPS_INT) & (pi < 1.0 - EPS_INT))[0]
        if len(active) == 0:
            break
        u = _null_vec(A[:, active])
        if u is None:
            break
        pi_act = pi[active]
        lp = lm = 1e15
        for pi_i, u_i in zip(pi_act, u):
            if   u_i >  EPS: lp = min(lp, (1.0 - pi_i) / u_i)
            elif u_i < -EPS: lp = min(lp, -pi_i / u_i)
            if  -u_i >  EPS: lm = min(lm, (1.0 - pi_i) / (-u_i))
            elif-u_i < -EPS: lm = min(lm,  pi_i / u_i)
        lp  = min(lp, 1e8); lm = min(lm, 1e8)
        tot = lp + lm
        if tot < EPS:
            break
        lam = lp if rng.random() < lm / tot else -lm
        pi[active] = np.clip(pi_act + lam * u, 0.0, 1.0)
        if track_phase:
            r_act = np.sqrt(np.maximum(pi_act, EPS))
            theta[active] += lam * u / r_act

    return pi, theta, int(np.sum((pi > EPS_INT) & (pi < 1.0 - EPS_INT)))


def landing_cube(pi: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Atterrissage Cube : greedy par distance à {0,1}."""
    pi = pi.copy()
    N  = len(pi)
    for _ in range(N + 20):
        active = np.where((pi > EPS_INT) & (pi < 1.0 - EPS_INT))[0]
        if len(active) == 0:
            break
        idx = active[int(np.argmin(np.minimum(pi[active], 1.0 - pi[active])))]
        pi[idx] = 1.0 if rng.random() < pi[idx] else 0.0
    return np.where(pi >= 0.5, 1.0, 0.0)


def landing_sphere(pi: np.ndarray, theta: np.ndarray,
                   Z: np.ndarray, pi_orig: np.ndarray,
                   rng: np.random.Generator,
                   alpha: float = 0.2) -> np.ndarray:
    """Atterrissage Sphère de Layla : balance-minimisant + cohérence de phase."""
    pi = pi.copy()
    N  = len(pi)

    def _bal_error(sel):
        if sel.sum() == 0:
            return np.inf
        pop_tot = Z.sum(axis=1)
        sel_tot = (Z[:, sel] / pi_orig[np.newaxis, sel]).sum(axis=1)
        return float(np.max(np.abs(sel_tot - pop_tot) / (np.abs(pop_tot) + EPS)))

    def _target_phase(active_set):
        amps = np.sqrt(np.maximum(pi[active_set], EPS))
        vec  = np.sum(amps * np.exp(1j * theta[active_set]))
        return float(np.angle(vec)) if abs(vec) > EPS else 0.0

    for _ in range(N + 20):
        active = np.where((pi > EPS_INT) & (pi < 1.0 - EPS_INT))[0]
        if len(active) == 0:
            break
        th_target  = _target_phase(active)
        best_idx   = None
        best_score = np.inf
        for j in active:
            pi_j_saved = pi[j]
            for b in [1.0, 0.0]:
                pi[j] = b
                err   = _bal_error(pi >= 1.0 - EPS_INT)
                align = np.cos(theta[j] - th_target)
                score = err - alpha * align * (1.0 if b == 1.0 else -1.0)
                if score < best_score:
                    best_score = score
                    best_idx   = j
            pi[j] = pi_j_saved
        if best_idx is None:
            best_idx = active[0]
        pi[best_idx] = 1.0 if rng.random() < pi[best_idx] else 0.0

    return np.where(pi >= 0.5, 1.0, 0.0)


def cube_sample(Z, pi, rng):
    A             = Z / np.where(pi > EPS, pi, EPS)[np.newaxis, :]
    pi_f, _, _    = flight_phase(pi, A, rng, track_phase=False)
    return landing_cube(pi_f, rng) > 0.5


def sphere_sample(Z, pi, rng, alpha=0.2):
    A                = Z / np.where(pi > EPS, pi, EPS)[np.newaxis, :]
    pi_f, theta, _   = flight_phase(pi, A, rng, track_phase=True)
    return landing_sphere(pi_f, theta, Z, pi, rng, alpha) > 0.5


# ============================================================
# GÉNÉRATION D'UNE POPULATION FIXE
# ============================================================

def generate_fixed_population(N: int, p: int, f: float,
                               seed: int = 2026,
                               pop_type: str = "correlated",
                               nonuniform_pi: bool = False,
                               continuous_y: bool = False) -> tuple:
    """
    Generates a FIXED population (fixed seed).
    Returns Z (p×N), y (N,), pi_orig (N,), p_true.

    nonuniform_pi : if True, pi_i are proportional to |Z[0,i]| + c,
                    creating rich variation in contrasts (y_i/pi_i - y_j/pi_j)^2.
                    If False, pi_i = n/N for all i (uniform, as in step01/02).

    continuous_y  : if True, y_i is the raw continuous score variable
                    (standardised). If False, y_i is binary (thresholded at p40).
                    Continuous y produces a full spectrum of contrast values,
                    making Q2 and Q4 much more informative.
    """
    rng = np.random.default_rng(seed)
    Z   = np.zeros((p, N))

    if pop_type == "mixed":
        for k in range(p):
            d = k % 4
            if   d == 0: Z[k] = rng.normal(0, 1, N)
            elif d == 1: Z[k] = rng.exponential(1, N)
            elif d == 2: Z[k] = rng.uniform(0, 10, N)
            else:        Z[k] = rng.binomial(5, 0.3, N).astype(float)
    elif pop_type == "skewed":
        Z = np.abs(rng.pareto(1.5, (p, N)))
    elif pop_type == "correlated":
        cov = 0.7 * np.ones((p, p)) + 0.3 * np.eye(p)
        L   = np.linalg.cholesky(cov)
        Z   = L @ rng.normal(0, 1, (p, N))

    score = Z[0] + 0.5 * Z[min(1, p - 1)] + rng.normal(0, 1, N)

    # ── Variable of interest y ──────────────────────────────
    if continuous_y:
        # Standardised continuous score — full spectrum of contrasts
        y      = (score - score.mean()) / (score.std() + EPS)
        p_true = float(y.mean())   # ≈ 0 by construction
    else:
        y      = (score > np.percentile(score, 40)).astype(float)
        p_true = float(y.mean())

    # ── Inclusion probabilities pi_i ────────────────────────
    n = max(p + 2, int(round(f * N)))

    if nonuniform_pi:
        # pi_i proportional to |Z[0,i]| + c (correlated with y),
        # bounded in [0.04, 0.96] and normalised to sum to n.
        raw = np.abs(Z[0]) + Z[0].std() * 0.5 + EPS
        pi  = raw / raw.sum() * n
        pi  = np.clip(pi, 0.04, 0.96)
        # Iterative rescaling to reach sum = n while respecting bounds
        for _ in range(20):
            s = pi.sum()
            if abs(s - n) < 1e-6:
                break
            pi = pi / s * n
            pi = np.clip(pi, 0.04, 0.96)
    else:
        pi_val = float(np.clip(n / N, 0.01, 0.99))
        pi     = np.full(N, pi_val)

    return Z, y, pi, p_true


# ============================================================
# MOTEUR D'ESTIMATION DES π_ij
# ============================================================

def estimate_pi_ij(Z: np.ndarray, y: np.ndarray,
                   pi_orig: np.ndarray,
                   R: int = 5000,
                   alpha: float = 0.2,
                   seed_base: int = 42,
                   verbose: bool = True) -> pd.DataFrame:
    """
    Estime π_ij^Cube et π_ij^Sphère pour toutes les paires (i<j)
    par simulation Monte Carlo sur R réplications.

    Population FIXÉE : Z, y, pi_orig ne changent pas entre réplications.
    Seule la source d'aléa (le générateur) varie.

    Retourne un DataFrame avec une ligne par paire (i,j) et les colonnes :
      i, j                     indices
      pi_i, pi_j               probabilités d'inclusion marginales
      pi_i_pi_j                produit (baseline indépendance)
      pi_ij_cube               estimation Monte Carlo, Cube
      pi_ij_sphere             estimation Monte Carlo, Sphère
      cov_cube                 π_ij^Cube  - π_i π_j
      cov_sphere               π_ij^Sphère - π_i π_j
      delta                    π_ij^Sphère - π_ij^Cube  (négatif = Sphère < Cube)
      contrast_sq              (y_i/π_i - y_j/π_j)²     (poids SYG)
      aux_dist_sq              Σ_k (A_ki - A_kj)²        (distance auxiliaire)
      syg_contrib_cube         terme SYG pour le Cube  (non-négatif)
      syg_contrib_sphere       terme SYG pour la Sphère
      syg_contrib_delta        différence des contributions SYG
    """
    N = len(pi_orig)
    n_pairs = N * (N - 1) // 2

    # Matrices de comptage des co-sélections
    count_cube   = np.zeros((N, N), dtype=np.float64)
    count_sphere = np.zeros((N, N), dtype=np.float64)

    t0 = time.time()
    for r in range(R):
        if verbose and (r % max(1, R // 20) == 0):
            pct = 100 * r / R
            ela = time.time() - t0
            eta = ela / (r + 1) * (R - r) / 60 if r > 0 else 0.0
            print(f"    Replication {r:5d}/{R}  ({pct:4.0f}%)  "
                  f"ETA: {eta:.1f}min", end="\r")

        # Cube
        rng_c = np.random.default_rng(seed_base + r)
        sel_c = cube_sample(Z, pi_orig, rng_c)
        idx_c = np.where(sel_c)[0]
        if len(idx_c) > 0:
            count_cube[np.ix_(idx_c, idx_c)] += 1.0

        # Sphère de Layla  (seed distinct pour indépendance)
        rng_s = np.random.default_rng(seed_base + R + r)
        sel_s = sphere_sample(Z, pi_orig, rng_s, alpha=alpha)
        idx_s = np.where(sel_s)[0]
        if len(idx_s) > 0:
            count_sphere[np.ix_(idx_s, idx_s)] += 1.0

    if verbose:
        print(f"    {R} replications completed in {time.time()-t0:.1f}s  " + " " * 20)

    # Matrice auxiliaire normalisée A[k,i] = z_ki / π_i
    A_mat = Z / np.where(pi_orig > EPS, pi_orig, EPS)[np.newaxis, :]

    # Construction du DataFrame par paire
    rows = []
    for i in range(N):
        for j in range(i + 1, N):
            pi_i  = float(pi_orig[i])
            pi_j  = float(pi_orig[j])
            indep = pi_i * pi_j

            pi_ij_c = count_cube[i, j]   / R
            pi_ij_s = count_sphere[i, j] / R

            cov_c = pi_ij_c - indep
            cov_s = pi_ij_s - indep
            delta = pi_ij_s - pi_ij_c

            # Contraste SYG : (y_i/π_i - y_j/π_j)²
            contrast = (y[i] / pi_i - y[j] / pi_j) ** 2

            # Distance auxiliaire : Σ_k (A_ki - A_kj)²
            aux_dist = float(np.sum((A_mat[:, i] - A_mat[:, j]) ** 2))

            # Contribution à la formule SYG (terme > 0 si cov < 0)
            syg_c = -(cov_c) * contrast      # positif ⟺ variance réduite
            syg_s = -(cov_s) * contrast
            syg_d = syg_s - syg_c            # négatif ⟺ Sphère < Cube (bon)

            rows.append({
                "i": i, "j": j,
                "pi_i": pi_i, "pi_j": pi_j, "pi_i_pi_j": indep,
                "pi_ij_cube":    pi_ij_c,
                "pi_ij_sphere":  pi_ij_s,
                "cov_cube":      cov_c,
                "cov_sphere":    cov_s,
                "delta":         delta,
                "contrast_sq":   contrast,
                "aux_dist_sq":   aux_dist,
                "syg_contrib_cube":   syg_c,
                "syg_contrib_sphere": syg_s,
                "syg_contrib_delta":  syg_d,
            })

    return pd.DataFrame(rows)


# ============================================================
# ANALYSE ET STATISTIQUES AGRÉGÉES
# ============================================================

def compute_summary(df: pd.DataFrame, config: dict) -> dict:
    """
    Calcule les statistiques agrégées pour les 5 questions.
    """
    n_pairs = len(df)

    # ── Q1 : fraction de paires où Sphère < Cube ──────────
    frac_lt  = float((df.delta < 0).mean())
    frac_eq  = float((df.delta == 0).mean())
    frac_gt  = float((df.delta > 0).mean())
    delta_mean = float(df.delta.mean())
    delta_median = float(df.delta.median())

    # ── Q2 : corrélation contraste × delta ─────────────────
    # On espère une corrélation NÉGATIVE :
    # grand contraste → delta très négatif (Sphère bien plus petite)
    corr_contrast_delta = float(np.corrcoef(
        df.contrast_sq, df.delta)[0, 1])

    # Quantiles du contraste : on prend les top 25% / 10% par rang
    # (tri explicite pour éviter les NaN avec distributions discrètes)
    df_sorted = df.sort_values("contrast_sq", ascending=False)
    n75 = max(1, int(len(df) * 0.25))
    n90 = max(1, int(len(df) * 0.10))
    top25 = df_sorted.iloc[:n75]
    top10 = df_sorted.iloc[:n90]
    delta_high_contrast_q75 = float(top25.delta.mean())
    delta_high_contrast_q90 = float(top10.delta.mean())
    frac_lt_high_contrast   = float((top25.delta < 0).mean())

    # ── Q3 : covariances par rapport à l'indépendance ──────
    cov_cube_mean   = float(df.cov_cube.mean())
    cov_sphere_mean = float(df.cov_sphere.mean())
    frac_neg_cov_cube   = float((df.cov_cube < 0).mean())
    frac_neg_cov_sphere = float((df.cov_sphere < 0).mean())

    # ── Q4 : distance auxiliaire × covariance Sphère ───────
    corr_aux_cov_sphere = float(np.corrcoef(
        df.aux_dist_sq, df.cov_sphere)[0, 1])
    corr_aux_cov_cube   = float(np.corrcoef(
        df.aux_dist_sq, df.cov_cube)[0, 1])

    # ── Q5 : variance SYG empirique ────────────────────────
    # Var_SYG = -Σ_{i<j} (π_ij - π_i π_j)(y_i/π_i - y_j/π_j)²
    #         = Σ_{i<j} syg_contrib  (avec le signe qu'on a choisi)
    var_syg_cube   = float(df.syg_contrib_cube.sum())
    var_syg_sphere = float(df.syg_contrib_sphere.sum())
    var_syg_reduction_pct = 100.0 * (var_syg_cube - var_syg_sphere) / (
        abs(var_syg_cube) + EPS)

    return {
        "config": config,
        "n_pairs": n_pairs,
        "Q1_fraction_sphere_lt_cube":    frac_lt,
        "Q1_fraction_sphere_eq_cube":    frac_eq,
        "Q1_fraction_sphere_gt_cube":    frac_gt,
        "Q1_delta_mean":                 delta_mean,
        "Q1_delta_median":               delta_median,
        "Q2_corr_contrast_delta":        corr_contrast_delta,
        "Q2_delta_mean_top25pct":        delta_high_contrast_q75,
        "Q2_delta_mean_top10pct":        delta_high_contrast_q90,
        "Q2_frac_lt_top25pct_contrast":  frac_lt_high_contrast,
        "Q3_cov_cube_mean":              cov_cube_mean,
        "Q3_cov_sphere_mean":            cov_sphere_mean,
        "Q3_frac_neg_cov_cube":          frac_neg_cov_cube,
        "Q3_frac_neg_cov_sphere":        frac_neg_cov_sphere,
        "Q4_corr_aux_dist_cov_sphere":   corr_aux_cov_sphere,
        "Q4_corr_aux_dist_cov_cube":     corr_aux_cov_cube,
        "Q5_var_SYG_cube":               var_syg_cube,
        "Q5_var_SYG_sphere":             var_syg_sphere,
        "Q5_var_SYG_reduction_pct":      var_syg_reduction_pct,
    }


def print_summary(s: dict):
    cfg = s["config"]
    print(f"\n{'='*65}")
    print(f"  RESULTS  N={cfg['N']}  p={cfg['p']}  f={cfg['f']}  "
          f"type={cfg['pop_type']}  alpha={cfg['alpha']}  R={cfg['R']}")
    print(f"  {s['n_pairs']} pairs (i,j) analysed")
    print(f"{'='*65}")

    print(f"\n  Q1 — Fraction of pairs where pi_ij^Sphere <= pi_ij^Cube:")
    print(f"    Sphere < Cube : {100*s['Q1_fraction_sphere_lt_cube']:5.1f}%")
    print(f"    Sphere = Cube : {100*s['Q1_fraction_sphere_eq_cube']:5.1f}%")
    print(f"    Sphere > Cube : {100*s['Q1_fraction_sphere_gt_cube']:5.1f}%")
    print(f"    Mean delta (S-C) : {s['Q1_delta_mean']:+.5f}")

    print(f"\n  Q2 — Contrast (y_i/pi_i - y_j/pi_j)^2 x delta:")
    print(f"    Corr(contrast, delta) : {s['Q2_corr_contrast_delta']:+.4f}")
    print(f"    [expected: negative — large contrasts -> more negative delta]")
    print(f"    Mean delta (top 25% contrasts) : "
          f"{s['Q2_delta_mean_top25pct']:+.5f}")
    print(f"    Mean delta (top 10% contrasts) : "
          f"{s['Q2_delta_mean_top10pct']:+.5f}")
    print(f"    % Sphere < Cube (top 25% contrast) : "
          f"{100*s['Q2_frac_lt_top25pct_contrast']:.1f}%")

    print(f"\n  Q3 — Covariances pi_ij - pi_i * pi_j:")
    print(f"    Cube   : mean = {s['Q3_cov_cube_mean']:+.5f}  "
          f"(% negative: {100*s['Q3_frac_neg_cov_cube']:.1f}%)")
    print(f"    Sphere : mean = {s['Q3_cov_sphere_mean']:+.5f}  "
          f"(% negative: {100*s['Q3_frac_neg_cov_sphere']:.1f}%)")

    print(f"\n  Q4 — Auxiliary distance sum(A_ki - A_kj)^2 vs covariance:")
    print(f"    Corr(aux.dist, cov.Cube)   : "
          f"{s['Q4_corr_aux_dist_cov_cube']:+.4f}")
    print(f"    Corr(aux.dist, cov.Sphere) : "
          f"{s['Q4_corr_aux_dist_cov_sphere']:+.4f}")
    print(f"    [expected Sphere: negative — distant pairs -> more negative cov]")

    print(f"\n  Q5 — Empirical SYG variance sum_{{i<j}} [-(pi_ij-pi_i*pi_j)(contrast)]:")
    print(f"    Cube   : {s['Q5_var_SYG_cube']:+.6f}")
    print(f"    Sphere : {s['Q5_var_SYG_sphere']:+.6f}")
    pct = s['Q5_var_SYG_reduction_pct']
    sign = "reduction" if pct > 0 else "increase"
    print(f"    Delta  : {pct:+.2f}%  ({sign})")
    print()


# ============================================================
# FIGURES
# ============================================================

def plot_figures(df: pd.DataFrame, summary: dict,
                 config_label: str, fig_dir: str):
    """
    Produit 4 figures pour les questions Q1–Q4.
    """
    if not HAS_MPL:
        return

    cfg = summary["config"]
    title_base = (f"N={cfg['N']}, p={cfg['p']}, "
                  f"f={cfg['f']}, {cfg['pop_type']}, alpha={cfg['alpha']}")

    # blue = Sphere < Cube (good), red = Sphere > Cube
    colors_delta = np.where(df.delta < 0, "#2166ac", "#d6604d")

    # ── Figure 1 : Q1 — scatter pi_ij^Sphere vs pi_ij^Cube ──
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(df.pi_ij_cube, df.pi_ij_sphere,
               c=colors_delta, alpha=0.4, s=8, linewidths=0)
    lim = max(df.pi_ij_cube.max(), df.pi_ij_sphere.max()) * 1.05
    ax.plot([0, lim], [0, lim], "k--", lw=0.8, label="y = x")
    pct_lt = 100 * summary["Q1_fraction_sphere_lt_cube"]
    ax.set_xlabel(r"$\hat{\pi}_{ij}^{\mathrm{Cube}}$", fontsize=12)
    ax.set_ylabel(r"$\hat{\pi}_{ij}^{\mathrm{Sphere}}$", fontsize=12)
    ax.set_title(f"Q1 — Pairwise $\\hat{{\\pi}}_{{ij}}$ compared\n{title_base}",
                 fontsize=10)
    ax.legend(fontsize=9)
    ax.text(0.05, 0.92,
            f"Sphere < Cube: {pct_lt:.1f}% of pairs",
            transform=ax.transAxes, fontsize=9,
            bbox=dict(facecolor="white", edgecolor="grey", alpha=0.8))
    fig.tight_layout()
    path = os.path.join(fig_dir, f"Q1_piij_scatter_{config_label}.pdf")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Figure Q1 -> {path}")

    # ── Figure 2 : Q2 — contrast x delta (key figure) ───────
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(df.contrast_sq, df.delta,
               c=colors_delta, alpha=0.35, s=8, linewidths=0)
    ax.axhline(0, color="black", lw=0.8, linestyle="--")

    # Trend line: bin means
    df_s = df.sort_values("contrast_sq")
    n_bins = 20
    bin_edges = np.percentile(df_s.contrast_sq, np.linspace(0, 100, n_bins + 1))
    bin_edges = np.unique(bin_edges)
    if len(bin_edges) > 2:
        bin_idx = np.digitize(df_s.contrast_sq, bin_edges[1:-1])
        bin_x, bin_y = [], []
        for b in range(len(bin_edges) - 1):
            mask = bin_idx == b
            if mask.sum() >= 3:
                bin_x.append(float(df_s.contrast_sq[mask].mean()))
                bin_y.append(float(df_s.delta[mask].mean()))
        if bin_x:
            ax.plot(bin_x, bin_y, "k-", lw=1.8, label="Bin mean")
            ax.legend(fontsize=9)

    corr = summary["Q2_corr_contrast_delta"]
    ax.set_xlabel(r"$(y_i/\pi_i - y_j/\pi_j)^2$  (SYG contrast)", fontsize=11)
    ax.set_ylabel(r"$\hat{\pi}_{ij}^{\mathrm{Sphere}} - \hat{\pi}_{ij}^{\mathrm{Cube}}$",
                  fontsize=11)
    ax.set_title(f"Q2 — Is the inequality concentrated on large contrasts?\n{title_base}",
                 fontsize=10)
    ax.text(0.97, 0.95,
            f"r = {corr:+.3f}",
            ha="right", va="top", transform=ax.transAxes, fontsize=10,
            bbox=dict(facecolor="white", edgecolor="grey", alpha=0.8))
    fig.tight_layout()
    path = os.path.join(fig_dir, f"Q2_contrast_delta_{config_label}.pdf")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Figure Q2 -> {path}")

    # ── Figure 3 : Q3 — covariance distributions ────────────
    fig, ax = plt.subplots(figsize=(7, 4))
    all_vals = np.concatenate([df.cov_cube, df.cov_sphere])
    bins = np.linspace(all_vals.min(), all_vals.max(), 60)
    ax.hist(df.cov_cube,   bins=bins, alpha=0.55, color="#d6604d",
            label=f"Cube   (mean={summary['Q3_cov_cube_mean']:+.4f})",
            edgecolor="none")
    ax.hist(df.cov_sphere, bins=bins, alpha=0.55, color="#2166ac",
            label=f"Sphere (mean={summary['Q3_cov_sphere_mean']:+.4f})",
            edgecolor="none")
    ax.axvline(0, color="black", lw=1.0, linestyle="--")
    ax.set_xlabel(r"$\hat{\pi}_{ij} - \pi_i\pi_j$  (empirical covariance)",
                  fontsize=11)
    ax.set_ylabel("Number of pairs", fontsize=11)
    ax.set_title(f"Q3 — Induced correlations vs independence\n{title_base}",
                 fontsize=10)
    ax.legend(fontsize=9)
    fig.tight_layout()
    path = os.path.join(fig_dir, f"Q3_covariances_{config_label}.pdf")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Figure Q3 -> {path}")

    # ── Figure 4 : Q4 — distance auxiliaire × covariance ───
    fig, ax = plt.subplots(figsize=(7, 5))
    col_cov = np.where(df.cov_sphere < 0, "#2166ac", "#d6604d")
    ax.scatter(df.aux_dist_sq, df.cov_sphere,
               c=col_cov, alpha=0.35, s=8, linewidths=0)
    ax.axhline(0, color="black", lw=0.8, linestyle="--")

    # Tendance par bin
    df_s2 = df.sort_values("aux_dist_sq")
    if len(df_s2) > 20:
        bin_edges2 = np.percentile(df_s2.aux_dist_sq,
                                   np.linspace(0, 100, 21))
        bin_edges2 = np.unique(bin_edges2)
        bin_idx2   = np.digitize(df_s2.aux_dist_sq, bin_edges2[1:-1])
        bx2, by2   = [], []
        for b in range(len(bin_edges2) - 1):
            mask = bin_idx2 == b
            if mask.sum() >= 3:
                bx2.append(float(df_s2.aux_dist_sq[mask].mean()))
                by2.append(float(df_s2.cov_sphere[mask].mean()))
        if bx2:
            ax.plot(bx2, by2, "k-", lw=1.8, label="Bin mean (Sphere)")
            ax.legend(fontsize=9)

    corr4 = summary["Q4_corr_aux_dist_cov_sphere"]
    ax.set_xlabel(r"$\sum_k (A_{ki} - A_{kj})^2$  (auxiliary distance)",
                  fontsize=11)
    ax.set_ylabel(r"$\hat{\pi}_{ij}^{\mathrm{Sphere}} - \pi_i\pi_j$",
                  fontsize=11)
    ax.set_title(f"Q4 — Does auxiliary distance predict co-selection?\n{title_base}",
                 fontsize=10)
    ax.text(0.97, 0.95,
            f"r = {corr4:+.3f}",
            ha="right", va="top", transform=ax.transAxes, fontsize=10,
            bbox=dict(facecolor="white", edgecolor="grey", alpha=0.8))
    fig.tight_layout()
    path = os.path.join(fig_dir, f"Q4_aux_dist_{config_label}.pdf")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Figure Q4 -> {path}")


# ============================================================
# UN RUN COMPLET (population + estimation + figures)
# ============================================================

def run_config(N: int, p: int, f: float, pop_type: str,
               alpha: float, R: int,
               pop_seed: int = 2026,
               est_seed: int = 42,
               nonuniform_pi: bool = False,
               continuous_y: bool = False,
               verbose: bool = True) -> dict:
    """
    Runs the full estimation for one configuration.
    Returns the summary dictionary.
    """
    design = ("nonuniform-pi+continuous-y" if (nonuniform_pi and continuous_y)
              else "nonuniform-pi" if nonuniform_pi
              else "continuous-y" if continuous_y
              else "uniform-pi+binary-y")
    config = dict(N=N, p=p, f=f, pop_type=pop_type, alpha=alpha, R=R,
                  design=design)
    suffix = ("_optB" if (nonuniform_pi or continuous_y) else "")
    label  = f"N{N}_p{p}_f{int(f*10)}__{pop_type}__a{int(alpha*10)}{suffix}"

    if verbose:
        print(f"\n{'─'*65}")
        print(f"  Config: N={N}  p={p}  f={f}  type={pop_type}"
              f"  alpha={alpha}  R={R}")
        print(f"  Design: {design}")
        print(f"{'─'*65}")

    # Fixed population
    Z, y, pi, p_true = generate_fixed_population(
        N, p, f, seed=pop_seed, pop_type=pop_type,
        nonuniform_pi=nonuniform_pi, continuous_y=continuous_y)
    n_pairs = N * (N - 1) // 2

    if verbose:
        print(f"  Fixed population (seed={pop_seed}): "
              f"N={N}, n={int(round(f*N))}, {n_pairs} pairs")
        print(f"  p_true = {p_true:.4f}")
        print(f"  Running {R} replications...")

    # Estimation des π_ij
    df_pairs = estimate_pi_ij(Z, y, pi, R=R, alpha=alpha,
                               seed_base=est_seed, verbose=verbose)

    # Résumé
    summary = compute_summary(df_pairs, config)
    if verbose:
        print_summary(summary)

    # Save CSV
    csv_path = os.path.join(OUT_DIR, f"pi_ij_pairs_{label}.csv")
    df_pairs.to_csv(csv_path, index=False, encoding="utf-8")
    if verbose:
        print(f"  CSV saved -> {csv_path}")

    # Figures
    if HAS_MPL:
        if verbose:
            print(f"  Generating figures...")
        plot_figures(df_pairs, summary, label, FIG_DIR)

    return summary


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    QUICK = "--test" in sys.argv
    FULL  = "--full" in sys.argv

    all_summaries = []

    if QUICK:
        # ── Test mode: N=30, R=500, single config ──────────
        print("\n  [TEST MODE] N=30, R=500")
        s = run_config(N=30, p=5, f=0.4,
                       pop_type="correlated", alpha=0.2,
                       R=500)
        all_summaries.append(s)

    elif FULL:
        # ── Full mode: multi-configuration grid ─────────────
        print("\n  [FULL MODE] Multi-configuration grid")
        from itertools import product as iprod
        grid = list(iprod(
            [50, 100],                  # N
            [2, 5, 10],                 # p
            [0.3, 0.5],                 # f
            ["mixed", "correlated"],    # pop_type
            [0.0, 0.2],                 # alpha
        ))
        print(f"  {len(grid)} configurations x R=2000 replications")
        for N, p, f, pt, alpha in grid:
            s = run_config(N=N, p=p, f=f, pop_type=pt,
                           alpha=alpha, R=2000)
            all_summaries.append(s)

    else:
        # ── Standard mode: 3 representative configurations ──
        print("\n  [STANDARD MODE]")

        # Config 1: hard case — large p, correlated population
        s1 = run_config(N=50, p=10, f=0.4,
                        pop_type="correlated", alpha=0.2,
                        R=5000)
        all_summaries.append(s1)

        # Config 2: medium case — p=5, mixed population
        s2 = run_config(N=50, p=5, f=0.4,
                        pop_type="mixed", alpha=0.2,
                        R=5000)
        all_summaries.append(s2)

        # Config 3: easy case — p=2 (sanity check)
        s3 = run_config(N=50, p=2, f=0.4,
                        pop_type="mixed", alpha=0.2,
                        R=5000)
        all_summaries.append(s3)

    # ── Option B mode: non-uniform pi + continuous y ────────
    # Always run after standard or QUICK mode (skip after FULL).
    # Non-uniform pi_i and continuous y produce a rich spectrum
    # of contrasts (y_i/pi_i - y_j/pi_j)^2, making Q2 and Q4
    # interpretable even at R=5000.
    if not FULL:
        print("\n  [OPTION B — non-uniform pi + continuous y]")

        # B1: hard case — p=10, correlated
        b1 = run_config(N=50, p=10, f=0.4,
                        pop_type="correlated", alpha=0.2,
                        R=5000,
                        nonuniform_pi=True, continuous_y=True)
        all_summaries.append(b1)

        # B2: medium case — p=5, mixed
        b2 = run_config(N=50, p=5, f=0.4,
                        pop_type="mixed", alpha=0.2,
                        R=5000,
                        nonuniform_pi=True, continuous_y=True)
        all_summaries.append(b2)

        # B3: easy case — p=2 (sanity check)
        b3 = run_config(N=50, p=2, f=0.4,
                        pop_type="mixed", alpha=0.2,
                        R=5000,
                        nonuniform_pi=True, continuous_y=True)
        all_summaries.append(b3)

    # ── Save global summary ─────────────────────────────────
    summary_path = os.path.join(OUT_DIR, "pi_ij_summary.json")
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(all_summaries, fh, indent=2, ensure_ascii=False,
                  default=float)
    print(f"\n  Global summary -> {summary_path}")

    # ── Summary table ───────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"  SUMMARY TABLE — pi_ij: CUBE vs LAYLA SPHERE")
    print(f"{'='*80}")
    hdr = (f"  {'Config':<35} {'%S<C':>6} {'r(ctr,d)':>10} "
           f"{'dSYG%':>8} {'cov.S':>8}")
    print(hdr)
    print(f"  {'─'*35} {'─'*6} {'─'*10} {'─'*8} {'─'*8}")
    for s in all_summaries:
        c  = s["config"]
        lbl = f"N={c['N']} p={c['p']} {c['pop_type']} a={c['alpha']}"
        pct_lt = 100 * s["Q1_fraction_sphere_lt_cube"]
        corr   = s["Q2_corr_contrast_delta"]
        syg    = s["Q5_var_SYG_reduction_pct"]
        cov_s  = s["Q3_cov_sphere_mean"]
        print(f"  {lbl:<35} {pct_lt:>5.1f}% {corr:>+10.3f} "
              f"{syg:>+7.2f}% {cov_s:>+8.5f}")
    print(f"{'='*80}\n")
