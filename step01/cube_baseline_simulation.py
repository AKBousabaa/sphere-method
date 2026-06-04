"""
=============================================================
  SIMULATION DE RÉFÉRENCE — BIAIS RÉSIDUEL DU CUBE
  Version validée — autonome (aucune dépendance externe)

  Auteurs :
    Bousabaa, A. & Sirolli, R. (2026)

  Référence théorique :
    Deville, J.-C. & Tillé, Y. (2004). Efficient balanced
    sampling: The cube method. Biometrika, 91(4), 893–912.

  Implémentation originale (SAS/INSEE, 1999) :
    Bousabaa, A., Lieber, J. & Sirolli, R.

  Objectif :
    Établir la baseline du Cube (biais résiduel d'atterrissage)
    que la Méthode de la Sphère devra améliorer.

  Métriques mesurées :
    - Erreur d'équilibrage résiduelle e_k  (après atterrissage)
    - Biais de l'estimateur HT : E[p̂_HT] - p_vrai
    - RMSE de p̂_HT
    - Nombre de valeurs fractionnaires résiduelles (n_frac)
    - % de tirages avec n_frac > 0

  Configurations simulées :
    N ∈ {50, 100, 200}          tailles de population
    p ∈ {2, 5, 10}              variables d'équilibrage
    f ∈ {0.2, 0.5, 0.8}        taux de sondage
    pop_type ∈ {mixed, skewed, correlated}
    R = 40 réplications par configuration
    Total : ~3 240 tirages Cube

  Résultats clés (à battre avec la Méthode de la Sphère) :
    balance_max_mean   = 4.3655
    bias_abs_mean      = 0.0572
    RMSE               = 0.0856
    pct_frac_gt0       = 100%
    n_frac_mean        = 5.67  (≈ p, conforme au théorème DT04)
=============================================================

Usage :
  python cube_baseline_simulation.py              # standard
  python cube_baseline_simulation.py --full       # grande simulation
  python cube_baseline_simulation.py --test       # 5 configs, vérification
"""

import numpy as np
import pandas as pd
import json
import time
import warnings
import sys
from itertools import product

warnings.filterwarnings("ignore")

# ============================================================
# CONSTANTES NUMÉRIQUES
# ============================================================

EPS     = 1e-10   # tolérance générale
EPS_INT = 1e-8    # tolérance pour détecter 0 ou 1


# ============================================================
# MÉTHODE DU CUBE — Deville & Tillé (2004)
# ============================================================

def _null_vec(A: np.ndarray):
    """
    Calcule le premier vecteur du noyau de A via SVD complète.
    A : matrice (p × n_active)
    Retourne : vecteur (n_active,) ou None si ker(A) trivial.
    """
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
        if norm < EPS:
            return None
        return u / norm
    except np.linalg.LinAlgError:
        return None


def cube_flight(pi: np.ndarray, A: np.ndarray,
                rng: np.random.Generator) -> tuple:
    """
    Phase de vol du Cube.
    Itère jusqu'à ce que le noyau de la sous-matrice active
    soit trivial (au plus p valeurs fractionnaires restantes).

    Paramètres
    ----------
    pi  : (N,) probabilités d'inclusion courantes
    A   : (p, N) matrice d'équilibrage A[k,i] = z_ki / π_i
    rng : générateur numpy

    Retourne
    --------
    pi  : (N,) vecteur mis à jour (beaucoup de 0 et 1)
    n_frac : nombre de valeurs encore fractionnaires
    """
    pi = pi.copy()
    N  = len(pi)

    for _ in range(500_000):
        # Ensemble actif
        active = np.where((pi > EPS_INT) & (pi < 1.0 - EPS_INT))[0]
        if len(active) == 0:
            break

        # Vecteur du noyau
        u = _null_vec(A[:, active])
        if u is None:
            break

        # Calcul des pas maximaux λ+ et λ-
        pi_act = pi[active]
        lp = lm = 1e15
        for pi_i, u_i in zip(pi_act, u):
            if   u_i >  EPS: lp = min(lp, (1.0 - pi_i) / u_i)
            elif u_i < -EPS: lp = min(lp, -pi_i / u_i)
            if  -u_i >  EPS: lm = min(lm, (1.0 - pi_i) / (-u_i))
            elif-u_i < -EPS: lm = min(lm,  pi_i / u_i)

        lp = min(lp, 1e8)
        lm = min(lm, 1e8)
        tot = lp + lm
        if tot < EPS:
            break

        # Mise à jour aléatoire (préserve E[π] = π)
        lam = lp if rng.random() < lm / tot else -lm
        pi[active] = np.clip(pi_act + lam * u, 0.0, 1.0)

    n_frac = int(np.sum((pi > EPS_INT) & (pi < 1.0 - EPS_INT)))
    return pi, n_frac


def cube_landing_greedy(pi: np.ndarray,
                        rng: np.random.Generator) -> np.ndarray:
    """
    Phase d'atterrissage greedy (implémentation Cube standard).
    Arrondit itérativement la valeur fractionnaire la plus proche
    de 0 ou 1, par tirage de Bernoulli.

    C'est cette phase qui introduit le biais résiduel
    que la Méthode de la Sphère cherche à réduire.
    """
    pi = pi.copy()
    N  = len(pi)

    for _ in range(N + 20):
        active = np.where((pi > EPS_INT) & (pi < 1.0 - EPS_INT))[0]
        if len(active) == 0:
            break
        # Greedy : le plus proche de 0 ou 1 en premier
        dists = np.minimum(pi[active], 1.0 - pi[active])
        idx   = active[int(np.argmin(dists))]
        pi[idx] = 1.0 if rng.random() < pi[idx] else 0.0

    return np.where(pi >= 0.5, 1.0, 0.0)


def cube_sample(Z: np.ndarray, pi: np.ndarray,
                rng: np.random.Generator) -> tuple:
    """
    Méthode du Cube complète (phase de vol + atterrissage).

    Paramètres
    ----------
    Z   : (p, N) variables auxiliaires brutes
    pi  : (N,)   probabilités d'inclusion ∈ (0, 1)
    rng : générateur numpy

    Retourne
    --------
    selected : (N,) indicatrice booléenne de l'échantillon
    n_frac   : nombre de valeurs fractionnaires à l'entrée
               de la phase d'atterrissage
    """
    # Matrice d'équilibrage : A[k, i] = z_ki / π_i
    A = Z / np.where(pi > EPS, pi, EPS)[np.newaxis, :]

    # Phase de vol
    pi_after, n_frac = cube_flight(pi, A, rng)

    # Phase d'atterrissage greedy
    pi_final = cube_landing_greedy(pi_after, rng)

    return pi_final > 0.5, n_frac


# ============================================================
# GÉNÉRATION DE POPULATIONS SIMULÉES
# ============================================================

def generate_population(N: int, p: int,
                         rng: np.random.Generator,
                         pop_type: str = "mixed") -> tuple:
    """
    Génère une population simulée de N unités avec p variables
    auxiliaires et une variable d'intérêt binaire y.

    pop_type :
      "mixed"      mélange réaliste de distributions
      "skewed"     distributions asymétriques (Pareto)
      "correlated" variables fortement corrélées (cas difficile
                   pour la SVD dans la phase de vol)

    Retourne
    --------
    Z      : (p, N) variables auxiliaires
    y      : (N,)   variable d'intérêt binaire
    p_true : proportion vraie de y=1
    """
    Z = np.zeros((p, N))

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

    # Variable d'intérêt : corrélée aux auxiliaires (simulation réaliste)
    score  = Z[0] + 0.5 * Z[min(1, p - 1)] + rng.normal(0, 1, N)
    y      = (score > np.percentile(score, 40)).astype(float)
    p_true = float(y.mean())

    return Z, y, p_true


# ============================================================
# UNE RÉPLICATION
# ============================================================

def run_replication(N: int, p: int, f: float,
                    pop_type: str, seed: int) -> dict:
    """
    Une réplication complète :
    génère une population, tire un échantillon Cube,
    calcule toutes les métriques.
    """
    rng = np.random.default_rng(seed)
    n   = max(p + 2, int(round(f * N)))
    if n >= N:
        return None

    Z, y, p_true = generate_population(N, p, rng, pop_type)

    pi_val = float(np.clip(n / N, 0.01, 0.99))
    pi     = np.full(N, pi_val)

    selected, n_frac = cube_sample(Z, pi, rng)
    n_sel = int(selected.sum())
    if n_sel == 0:
        return None

    # Erreurs d'équilibrage (après atterrissage)
    pop_totals = Z.sum(axis=1)
    sel_totals = (Z[:, selected] / pi[np.newaxis, selected]).sum(axis=1)
    denom      = np.abs(pop_totals) + EPS
    errors     = np.abs(sel_totals - pop_totals) / denom

    # Estimateur Horvitz-Thompson
    p_hat = float((y[selected] / pi[selected]).sum() / N)
    bias  = p_hat - p_true

    return {
        "N":        N,
        "p":        p,
        "f":        f,
        "pop_type": pop_type,
        "n_sel":    n_sel,
        "n_exp":    n,
        "n_frac":   n_frac,
        "e_max":    float(errors.max()),
        "e_mean":   float(errors.mean()),
        "p_hat":    p_hat,
        "p_true":   p_true,
        "bias":     bias,
    }


# ============================================================
# SIMULATION PRINCIPALE
# ============================================================

def run_simulation(N_values, p_values, f_values, pop_types,
                   R: int = 40, verbose: bool = True) -> pd.DataFrame:
    """
    Lance la simulation complète.

    Paramètres
    ----------
    N_values  : liste de tailles de population
    p_values  : liste de nombres de variables d'équilibrage
    f_values  : liste de taux de sondage
    pop_types : liste de types de population
    R         : réplications par configuration
    verbose   : affichage de la progression
    """
    configs = list(product(N_values, p_values, f_values, pop_types))
    total   = len(configs) * R

    if verbose:
        print("=" * 62)
        print("  SIMULATION BASELINE — MÉTHODE DU CUBE")
        print("  Deville & Tillé (2004)")
        print("  Bousabaa, Lieber & Sirolli (INSEE, 1999)")
        print("=" * 62)
        print(f"  Configurations : {len(configs)}")
        print(f"  Réplications   : {R} par configuration")
        print(f"  Total estimé   : ~{total:,} tirages Cube")
        print("=" * 62)

    rows = []
    t0   = time.time()
    done = 0

    for ci, (N, p, f, pt) in enumerate(configs):
        n_try = max(p + 2, int(round(f * N)))
        if n_try >= N:
            if verbose:
                print(f"  Skip N={N} p={p} f={f:.1f} (n={n_try} ≥ N)")
            continue

        if verbose:
            elapsed = time.time() - t0
            rate    = done / elapsed if (elapsed > 0 and done > 0) else None
            eta     = (total - done) / rate / 60 if rate else 0.0
            print(f"  [{ci+1:3d}/{len(configs)}] N={N:5d} p={p:2d} "
                  f"f={f:.1f} {pt:<12} R={R}  ETA:{eta:.1f}min",
                  end="\r")

        for rep in range(R):
            seed   = 99991 * ci + 7 * rep + 13
            result = run_replication(N, p, f, pt, seed)
            if result is not None:
                rows.append(result)
        done += R

    elapsed = time.time() - t0
    df = pd.DataFrame(rows)
    if verbose:
        print(f"\n\n  {len(df):,} réplications valides en {elapsed:.1f}s")
    return df


# ============================================================
# RAPPORT ET ANALYSE
# ============================================================

def print_report(df: pd.DataFrame):
    """Rapport structuré des résultats."""

    print("\n" + "=" * 62)
    print("  RAPPORT — BIAIS RÉSIDUEL DU CUBE (BASELINE)")
    print("=" * 62)

    # Erreur d'équilibrage par N
    print("\n  [1] Erreur d'équilibrage max (e_max) par N :")
    for N, g in df.groupby("N"):
        em  = g.e_max.mean()
        es  = g.e_max.std()
        p95 = float(np.percentile(g.e_max, 95))
        ba  = g.bias.abs().mean()
        rm  = float(np.sqrt((g.bias**2).mean()))
        nf  = g.n_frac.mean()
        pnf = (g.n_frac > 0).mean() * 100
        print(f"    N={N:5d}  e_max={em:.4f}±{es:.4f}  p95={p95:.4f}"
              f"  |biais|={ba:.5f}  RMSE={rm:.5f}"
              f"  frac_moy={nf:.2f}  frac>0:{pnf:.0f}%")

    # Erreur d'équilibrage par p
    print("\n  [2] Erreur d'équilibrage max par p :")
    for p, g in df.groupby("p"):
        em  = g.e_max.mean()
        ba  = g.bias.abs().mean()
        rm  = float(np.sqrt((g.bias**2).mean()))
        pnf = (g.n_frac > 0).mean() * 100
        nf  = g.n_frac.mean()
        print(f"    p={p:2d}  e_max={em:.4f}"
              f"  |biais|={ba:.5f}  RMSE={rm:.5f}"
              f"  frac_moy={nf:.2f}  frac>0:{pnf:.0f}%")

    # Erreur par taux de sondage
    print("\n  [3] Erreur d'équilibrage max par f :")
    for f, g in df.groupby("f"):
        em = g.e_max.mean()
        ba = g.bias.abs().mean()
        nf = g.n_frac.mean()
        print(f"    f={f:.1f}  e_max={em:.4f}"
              f"  |biais|={ba:.5f}  frac_moy={nf:.2f}")

    # Erreur par type de population
    print("\n  [4] Erreur d'équilibrage max par type de population :")
    for pt, g in df.groupby("pop_type"):
        em  = g.e_max.mean()
        ba  = g.bias.abs().mean()
        pnf = (g.n_frac > 0).mean() * 100
        print(f"    {pt:<12}  e_max={em:.4f}"
              f"  |biais|={ba:.5f}  frac>0:{pnf:.0f}%")

    # Métriques globales
    print("\n" + "─" * 62)
    print("  MÉTRIQUES GLOBALES BASELINE")
    print("  (valeurs de référence pour la Méthode de la Sphère)")
    print("─" * 62)
    metrics = {
        "balance_max_mean":        float(df.e_max.mean()),
        "balance_mean_mean":       float(df.e_mean.mean()),
        "bias_abs_mean":           float(df.bias.abs().mean()),
        "rmse":                    float(np.sqrt((df.bias**2).mean())),
        "pct_frac_gt0":            float((df.n_frac > 0).mean()),
        "n_frac_mean_when_gt0":    float(df[df.n_frac > 0].n_frac.mean())
                                   if (df.n_frac > 0).any() else 0.0,
    }
    for k, v in metrics.items():
        print(f"    {k:<30} : {v:.6f}")
    print("─" * 62)

    return metrics


# ============================================================
# EXPORT
# ============================================================

def save_results(df: pd.DataFrame, metrics: dict,
                 prefix: str = "cube_baseline"):
    """Sauvegarde CSV + JSON."""
    csv_path  = f"/mnt/user-data/outputs/{prefix}_raw.csv"
    meta_path = f"/mnt/user-data/outputs/{prefix}_meta.json"

    df.to_csv(csv_path, index=False, encoding="utf-8")

    meta = {
        "method":     "Cube (Deville & Tillé 2004)",
        "date":       pd.Timestamp.now().isoformat(),
        "n_results":  len(df),
        "n_configs":  df.groupby(["N", "p", "f", "pop_type"]).ngroups,
        "key_metrics": {k: round(v, 6) for k, v in metrics.items()},
        "note": (
            "Ces métriques constituent la baseline à améliorer "
            "avec la Méthode de la Sphère (en cours de développement)."
        ),
    }
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, ensure_ascii=False)

    print(f"\n  Fichiers sauvegardés :")
    print(f"    {csv_path}   ({len(df):,} lignes)")
    print(f"    {meta_path}")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    QUICK = "--test" in sys.argv
    FULL  = "--full" in sys.argv

    if QUICK:
        # Vérification rapide (~30s)
        df = run_simulation(
            N_values  = [50, 100],
            p_values  = [2, 5],
            f_values  = [0.3, 0.7],
            pop_types = ["mixed"],
            R         = 10,
        )
    elif FULL:
        # Simulation complète (~30 min)
        df = run_simulation(
            N_values  = [50, 100, 200, 500, 1_000, 5_000],
            p_values  = [2, 5, 10, 20],
            f_values  = [0.1, 0.2, 0.3, 0.5, 0.7, 0.8],
            pop_types = ["mixed", "normal", "skewed", "correlated"],
            R         = 200,
        )
    else:
        # Standard (~60s)
        df = run_simulation(
            N_values  = [50, 100, 200],
            p_values  = [2, 5, 10],
            f_values  = [0.2, 0.5, 0.8],
            pop_types = ["mixed", "skewed", "correlated"],
            R         = 40,
        )

    metrics = print_report(df)
    save_results(df, metrics)
