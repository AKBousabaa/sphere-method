"""
=============================================================
  MÉTHODE DE LA SPHÈRE — Implémentation et comparaison
  avec la Méthode du Cube

  Idée fondatrice :
    Représenter les probabilités d'inclusion comme amplitudes
    complexes ψ_i = √π_i · e^{iθ_i} sur la sphère unité
    dans C^N (définie par Σ|ψ_i|² = n).

    La phase θ_i s'accumule durant la phase de vol et porte
    une mémoire de la trajectoire de chaque unité.
    La phase d'atterrissage exploite cette information pour
    un arrondi collectif minimisant l'erreur d'équilibrage.

  Différences clés vs. Cube (Deville & Tillé 2004) :
    - Phase de vol    : IDENTIQUE (même martingale sur π)
                        + tracking de la phase θ
    - Phase d'atterrissage :
        Cube    → greedy "plus proche de 0 ou 1" (indépendant)
        Sphère  → greedy "balance-minimisant" avec la phase
                  comme critère de bris d'égalité

  Note sur le tunneling :
    Si |ψ_i|² dépasse transitoirement [0,1] (ce qui ne se produit
    pas en pratique avec des niveaux d'énergie équivalents),
    la projection sur la sphère ramènerait naturellement la valeur
    dans le domaine admissible.

  Référence baseline :
    Cube (Deville & Tillé 2004) — voir cube_baseline_simulation.py

  Auteurs : Bousabaa, A. & Sirolli, R. (2026)
=============================================================
"""

import numpy as np
import pandas as pd
import json
import time
import warnings
import sys

warnings.filterwarnings("ignore")

EPS     = 1e-10
EPS_INT = 1e-8


# ============================================================
# UTILITAIRES COMMUNS
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


def generate_population(N: int, p: int,
                         rng: np.random.Generator,
                         pop_type: str = "mixed") -> tuple:
    """Population simulée (identique à la baseline)."""
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
    score  = Z[0] + 0.5 * Z[min(1, p - 1)] + rng.normal(0, 1, N)
    y      = (score > np.percentile(score, 40)).astype(float)
    return Z, y, float(y.mean())


def compute_balance_error(selected: np.ndarray,
                           Z: np.ndarray,
                           pi: np.ndarray) -> float:
    """
    Erreur d'équilibrage max (relative) pour une sélection donnée.
    Σ_{i∈S} z_ki/π_i  vs.  Σ_{i∈U} z_ki
    """
    if selected.sum() == 0:
        return np.inf
    pop_tot = Z.sum(axis=1)
    sel_tot = (Z[:, selected] / pi[np.newaxis, selected]).sum(axis=1)
    denom   = np.abs(pop_tot) + EPS
    return float(np.max(np.abs(sel_tot - pop_tot) / denom))


# ============================================================
# PHASE DE VOL COMMUNE (Cube et Sphère)
# ============================================================

def flight_phase(pi: np.ndarray, A: np.ndarray,
                 rng: np.random.Generator,
                 track_phase: bool = False) -> tuple:
    """
    Phase de vol du Cube / de la Sphère.

    Paramètres
    ----------
    track_phase : si True, calcule et retourne les phases θ_i

    Retourne
    --------
    pi    : (N,) après la phase de vol
    theta : (N,) phases accumulées (None si track_phase=False)
    n_frac: nombre de valeurs fractionnaires résiduelles
    """
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
            elif-u_i < -EPS: lm = min(lm, pi_i / u_i)

        lp  = min(lp, 1e8)
        lm  = min(lm, 1e8)
        tot = lp + lm
        if tot < EPS:
            break

        lam = lp if rng.random() < lm / tot else -lm

        # Mise à jour des π (identique Cube et Sphère)
        pi[active] = np.clip(pi_act + lam * u, 0.0, 1.0)

        # Mise à jour des phases θ (Sphère uniquement)
        # θ_i accumule la vitesse angulaire relative de chaque unité
        # = variation de π_i / amplitude courante √π_i
        if track_phase:
            r_act = np.sqrt(np.maximum(pi_act, EPS))
            theta[active] += lam * u / r_act

    n_frac = int(np.sum((pi > EPS_INT) & (pi < 1.0 - EPS_INT)))
    return pi, theta, n_frac


# ============================================================
# PHASE D'ATTERRISSAGE — CUBE (greedy classique)
# ============================================================

def landing_cube(pi: np.ndarray,
                 rng: np.random.Generator) -> np.ndarray:
    """
    Atterrissage Cube : arrondit le plus proche de 0 ou 1,
    stochastiquement. N'utilise pas l'information sur l'équilibrage.
    """
    pi = pi.copy()
    N  = len(pi)
    for _ in range(N + 20):
        active = np.where((pi > EPS_INT) & (pi < 1.0 - EPS_INT))[0]
        if len(active) == 0:
            break
        dists = np.minimum(pi[active], 1.0 - pi[active])
        idx   = active[int(np.argmin(dists))]
        pi[idx] = 1.0 if rng.random() < pi[idx] else 0.0
    return np.where(pi >= 0.5, 1.0, 0.0)


# ============================================================
# PHASE D'ATTERRISSAGE — SPHÈRE (balance-minimisant + phase)
# ============================================================

def landing_sphere(pi: np.ndarray, theta: np.ndarray,
                   Z: np.ndarray, pi_orig: np.ndarray,
                   rng: np.random.Generator,
                   alpha: float = 0.2) -> np.ndarray:
    """
    Atterrissage de la Méthode de la Sphère.

    À chaque étape, on choisit l'unité dont l'arrondi (dans la
    direction optimale) minimise l'erreur d'équilibrage résiduelle.
    La phase θ_i entre comme critère de bris d'égalité pondéré
    (poids alpha).

    Paramètres
    ----------
    pi       : (N,) probabilités après la phase de vol
    theta    : (N,) phases accumulées durant la phase de vol
    Z        : (p, N) variables auxiliaires
    pi_orig  : (N,) probabilités d'inclusion originales (pour HT)
    rng      : générateur
    alpha    : poids de la correction de phase (0 = Sphère pure
               sans phase, >0 = correction cohérence)

    L'arrondi reste stochastique avec P(round_i = 1) = π_i,
    ce qui préserve la propriété de sans-biais de l'estimateur HT.
    """
    pi    = pi.copy()
    N     = len(pi)
    p_var = Z.shape[0]

    # Phase cible : argument de la somme des amplitudes actives
    # (analogue au "groupe de référence" en mécanique quantique)
    def target_phase(active_set):
        amps = np.sqrt(np.maximum(pi[active_set], EPS))
        th   = theta[active_set]
        vec  = np.sum(amps * np.exp(1j * th))
        return float(np.angle(vec)) if abs(vec) > EPS else 0.0

    for _ in range(N + 20):
        active = np.where((pi > EPS_INT) & (pi < 1.0 - EPS_INT))[0]
        if len(active) == 0:
            break

        # Sélection déjà décidée (π = 0 ou 1)
        decided_sel = pi >= 1.0 - EPS_INT

        # Phase de référence du groupe actif
        th_target = target_phase(active)

        best_idx   = None
        best_score = np.inf

        for j in active:
            # Simuler l'arrondi de j à 1 et à 0
            pi_j_saved = pi[j]

            for b in [1.0, 0.0]:
                pi[j] = b
                sel_test = pi >= 1.0 - EPS_INT
                err = compute_balance_error(sel_test, Z, pi_orig)

                # Correction de cohérence de phase
                # Un arrondi aligné avec le groupe (cos > 0) est préféré
                th_j    = theta[j]
                align   = np.cos(th_j - th_target)
                # Si b=1 : aligner encourage la sélection (score réduit)
                # Si b=0 : anti-aligner encourage l'exclusion
                phase_bonus = alpha * align * (1.0 if b == 1.0 else -1.0)
                score = err - phase_bonus

                if score < best_score:
                    best_score = score
                    best_idx   = j

            pi[j] = pi_j_saved

        if best_idx is None:
            best_idx = active[0]

        # Arrondi stochastique : P(round=1) = π_i (sans-biais préservé)
        pi[best_idx] = 1.0 if rng.random() < pi[best_idx] else 0.0

    return np.where(pi >= 0.5, 1.0, 0.0)


# ============================================================
# MÉTHODES COMPLÈTES
# ============================================================

def cube_sample(Z: np.ndarray, pi: np.ndarray,
                rng: np.random.Generator) -> tuple:
    """Méthode du Cube complète."""
    A              = Z / np.where(pi > EPS, pi, EPS)[np.newaxis, :]
    pi_f, _, nf   = flight_phase(pi, A, rng, track_phase=False)
    pi_final       = landing_cube(pi_f, rng)
    return pi_final > 0.5, nf


def sphere_sample(Z: np.ndarray, pi: np.ndarray,
                  rng: np.random.Generator,
                  alpha: float = 0.2) -> tuple:
    """Méthode de la Sphère complète."""
    A                  = Z / np.where(pi > EPS, pi, EPS)[np.newaxis, :]
    pi_f, theta, nf   = flight_phase(pi, A, rng, track_phase=True)
    pi_final           = landing_sphere(pi_f, theta, Z, pi, rng, alpha)
    return pi_final > 0.5, nf


# ============================================================
# RÉPLICATION COMPARATIVE
# ============================================================

def run_comparison(N: int, p: int, f: float,
                   pop_type: str, seed: int,
                   alpha: float = 0.2) -> dict:
    """
    Compare Cube vs Sphère sur la même population et le même seed.
    Garantit que les deux méthodes affrontent exactement le même
    problème pour une comparaison équitable.
    """
    n = max(p + 2, int(round(f * N)))
    if n >= N:
        return None

    # Même population pour les deux méthodes
    rng_pop = np.random.default_rng(seed)
    Z, y, p_true = generate_population(N, p, rng_pop, pop_type)
    pi_val = float(np.clip(n / N, 0.01, 0.99))
    pi     = np.full(N, pi_val)

    # ---- CUBE ----
    rng_c  = np.random.default_rng(seed + 1_000_000)
    sel_c, nf_c = cube_sample(Z, pi, rng_c)
    ns_c = int(sel_c.sum())
    if ns_c == 0:
        return None
    pop_tot = Z.sum(axis=1)
    st_c    = (Z[:, sel_c] / pi[np.newaxis, sel_c]).sum(axis=1)
    err_c   = np.abs(st_c - pop_tot) / (np.abs(pop_tot) + EPS)
    phat_c  = float((y[sel_c] / pi[sel_c]).sum() / N)

    # ---- SPHÈRE ----
    rng_s  = np.random.default_rng(seed + 2_000_000)
    sel_s, nf_s = sphere_sample(Z, pi, rng_s, alpha=alpha)
    ns_s = int(sel_s.sum())
    if ns_s == 0:
        return None
    st_s  = (Z[:, sel_s] / pi[np.newaxis, sel_s]).sum(axis=1)
    err_s = np.abs(st_s - pop_tot) / (np.abs(pop_tot) + EPS)
    phat_s = float((y[sel_s] / pi[sel_s]).sum() / N)

    return {
        # Config
        "N": N, "p": p, "f": f, "pop_type": pop_type, "alpha": alpha,
        "p_true": p_true,
        # Cube
        "cube_e_max":  float(err_c.max()),
        "cube_e_mean": float(err_c.mean()),
        "cube_bias":   phat_c - p_true,
        "cube_nfrac":  nf_c,
        "cube_nsel":   ns_c,
        # Sphère
        "sph_e_max":   float(err_s.max()),
        "sph_e_mean":  float(err_s.mean()),
        "sph_bias":    phat_s - p_true,
        "sph_nfrac":   nf_s,
        "sph_nsel":    ns_s,
        # Delta (Sphère - Cube : négatif = amélioration)
        "delta_e_max":  float(err_s.max())  - float(err_c.max()),
        "delta_e_mean": float(err_s.mean()) - float(err_c.mean()),
        "delta_bias":   abs(phat_s - p_true) - abs(phat_c - p_true),
    }


# ============================================================
# SIMULATION COMPARATIVE
# ============================================================

def run_simulation(N_values, p_values, f_values, pop_types,
                   alpha_values, R: int = 40,
                   verbose: bool = True) -> pd.DataFrame:
    """
    Simulation comparative Cube vs Sphère sur toutes les
    configurations.
    """
    from itertools import product
    configs = list(product(N_values, p_values, f_values, pop_types,
                           alpha_values))

    if verbose:
        print("=" * 62)
        print("  SIMULATION COMPARATIVE : CUBE vs SPHÈRE")
        print("=" * 62)
        print(f"  Configurations  : {len(configs)}")
        print(f"  Réplications    : {R} par configuration")
        print(f"  Total estimé    : ~{len(configs)*R:,} paires Cube/Sphère")
        print("=" * 62)

    rows = []
    t0   = time.time()
    done = 0
    total = len(configs) * R

    for ci, (N, p, f, pt, alpha) in enumerate(configs):
        if max(p + 2, int(round(f * N))) >= N:
            continue

        if verbose:
            elapsed = time.time() - t0
            rate    = done / elapsed if (elapsed > 0 and done > 0) else None
            eta     = (total - done) / rate / 60 if rate else 0.0
            print(f"  [{ci+1:3d}/{len(configs)}] N={N:4d} p={p:2d} "
                  f"f={f:.1f} {pt:<12} α={alpha:.2f} "
                  f"ETA:{eta:.1f}min", end="\r")

        for rep in range(R):
            seed   = 13337 * ci + 997 * rep
            result = run_comparison(N, p, f, pt, seed, alpha)
            if result is not None:
                result["rep"] = rep
                rows.append(result)
        done += R

    elapsed = time.time() - t0
    df = pd.DataFrame(rows)
    if verbose:
        print(f"\n\n  {len(df):,} paires valides en {elapsed:.1f}s")
    return df


# ============================================================
# RAPPORT COMPARATIF
# ============================================================

def print_report(df: pd.DataFrame):
    """
    Rapport structuré : Cube vs Sphère.
    Métriques clés pour l'article.
    """
    print("\n" + "=" * 62)
    print("  RAPPORT COMPARATIF — CUBE vs SPHÈRE")
    print("=" * 62)

    # Fonction utilitaire
    def arrow(delta, reverse=False):
        """↑ si amélioration (delta < 0 normalement)."""
        improved = delta < 0 if not reverse else delta > 0
        return "↓ mieux" if improved else "↑ moins bien"

    # Global
    print("\n  [A] RÉSULTATS GLOBAUX (toutes configurations)")
    print("─" * 62)
    print(f"  {'Métrique':<30} {'Cube':>10} {'Sphère':>10} {'Δ':>10}")
    print(f"  {'─'*30} {'─'*10} {'─'*10} {'─'*10}")

    metrics = [
        ("e_max moyen",  "cube_e_max",  "sph_e_max"),
        ("e_mean moyen", "cube_e_mean", "sph_e_mean"),
        ("|biais| moyen","cube_bias",   "sph_bias"),
        ("RMSE",         None,          None),
    ]

    e_max_c  = df.cube_e_max.mean()
    e_max_s  = df.sph_e_max.mean()
    e_mean_c = df.cube_e_mean.mean()
    e_mean_s = df.sph_e_mean.mean()
    bias_c   = df.cube_bias.abs().mean()
    bias_s   = df.sph_bias.abs().mean()
    rmse_c   = float(np.sqrt((df.cube_bias**2).mean()))
    rmse_s   = float(np.sqrt((df.sph_bias**2).mean()))

    for label, vc, vs in [
        ("e_max moyen",   e_max_c, e_max_s),
        ("e_mean moyen",  e_mean_c, e_mean_s),
        ("|biais| moyen", bias_c,  bias_s),
        ("RMSE",          rmse_c,  rmse_s),
    ]:
        d = vs - vc
        pct = 100 * d / (vc + EPS)
        sign = "↓" if d < 0 else "↑"
        print(f"  {label:<30} {vc:>10.5f} {vs:>10.5f} "
              f"{sign}{abs(pct):>6.1f}%")

    # Par N
    print("\n  [B] e_max moyen par N :")
    for N, g in df.groupby("N"):
        c = g.cube_e_max.mean()
        s = g.sph_e_max.mean()
        d = s - c
        pct = 100*d/(c+EPS)
        sign = "↓" if d < 0 else "↑"
        print(f"    N={N:5d}  Cube={c:.4f}  Sphère={s:.4f}  "
              f"{sign}{abs(pct):.1f}%")

    # Par p
    print("\n  [C] e_max moyen par p :")
    for p, g in df.groupby("p"):
        c = g.cube_e_max.mean()
        s = g.sph_e_max.mean()
        d = s - c
        pct = 100*d/(c+EPS)
        sign = "↓" if d < 0 else "↑"
        print(f"    p={p:2d}  Cube={c:.4f}  Sphère={s:.4f}  "
              f"{sign}{abs(pct):.1f}%")

    # Par type
    print("\n  [D] e_max moyen par type de population :")
    for pt, g in df.groupby("pop_type"):
        c = g.cube_e_max.mean()
        s = g.sph_e_max.mean()
        d = s - c
        pct = 100*d/(c+EPS)
        sign = "↓" if d < 0 else "↑"
        print(f"    {pt:<12}  Cube={c:.4f}  Sphère={s:.4f}  "
              f"{sign}{abs(pct):.1f}%")

    # Par alpha
    if df.alpha.nunique() > 1:
        print("\n  [E] Sensibilité au paramètre α (poids de phase) :")
        for alpha, g in df.groupby("alpha"):
            c = g.cube_e_max.mean()
            s = g.sph_e_max.mean()
            pct = 100*(s-c)/(c+EPS)
            sign = "↓" if s < c else "↑"
            print(f"    α={alpha:.2f}  Cube={c:.4f}  Sphère={s:.4f}  "
                  f"{sign}{abs(pct):.1f}%")

    # % cas où la Sphère bat le Cube
    better_e = (df.delta_e_max < 0).mean() * 100
    better_b = (df.delta_bias < 0).mean() * 100
    print(f"\n  [F] % cas où la Sphère améliore le Cube :")
    print(f"    e_max réduit  : {better_e:.1f}% des réplications")
    print(f"    |biais| réduit: {better_b:.1f}% des réplications")

    print("\n" + "=" * 62)

    return {
        "cube": {"e_max": e_max_c, "e_mean": e_mean_c,
                 "bias_abs": bias_c, "rmse": rmse_c},
        "sphere": {"e_max": e_max_s, "e_mean": e_mean_s,
                   "bias_abs": bias_s, "rmse": rmse_s},
        "pct_better_e_max": float(better_e),
        "pct_better_bias":  float(better_b),
    }


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    QUICK = "--test"  in sys.argv
    FULL  = "--full"  in sys.argv

    if QUICK:
        df = run_simulation(
            N_values    = [50, 100],
            p_values    = [2, 5],
            f_values    = [0.3, 0.7],
            pop_types   = ["mixed"],
            alpha_values= [0.0, 0.2],
            R           = 10,
        )
    elif FULL:
        df = run_simulation(
            N_values    = [50, 100, 200, 500],
            p_values    = [2, 5, 10, 20],
            f_values    = [0.1, 0.2, 0.3, 0.5, 0.7, 0.8],
            pop_types   = ["mixed", "skewed", "correlated"],
            alpha_values= [0.0, 0.1, 0.2, 0.4],
            R           = 100,
        )
    else:
        # Standard
        df = run_simulation(
            N_values    = [50, 100, 200],
            p_values    = [2, 5, 10],
            f_values    = [0.2, 0.5, 0.8],
            pop_types   = ["mixed", "skewed", "correlated"],
            alpha_values= [0.0, 0.2, 0.4],
            R           = 30,
        )

    summary = print_report(df)

    # Sauvegarde
    df.to_csv("/mnt/user-data/outputs/sphere_vs_cube_raw.csv",
              index=False, encoding="utf-8")
    meta = {
        "date":    pd.Timestamp.now().isoformat(),
        "methods": ["Cube (Deville & Tillé 2004)",
                    "Sphere (Bousabaa 2026)"],
        "n_results": len(df),
        "summary": summary,
    }
    with open("/mnt/user-data/outputs/sphere_vs_cube_meta.json",
              "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"\n  Fichiers sauvegardés :")
    print(f"    sphere_vs_cube_raw.csv  ({len(df):,} lignes)")
    print(f"    sphere_vs_cube_meta.json")
