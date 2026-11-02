"""Robust statistics and tail-risk metrics for the paper (reviewer points on statistics and tails).

Uses only results already on disk:
  results/test_episodes.csv        one row per controller x seed x test file (validation-best models)
  results/test_final_episodes.csv  same for the final models (python evaluate.py --model final --rl-only)

Reports
  1. per scenario x controller: mean with a bootstrap 95% CI over seeds, median, 95th percentile and
     CVaR (mean of the worst 10% of episodes) of the episode waiting time, failure probability (Wilson CI)
  2. per demand level: median episode waiting time and failures
  3. paired Wilcoxon signed-rank tests on identical test files (per-file mean over seeds), Holm-corrected
  4. ablation: Fisher's exact test on failed episodes and on seeds with at least one failure
  5. checkpoint selection: final vs. validation-best model

Run from the repository root:  python paper/robust_stats.py
"""
import os
import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
SCEN = ["cologne1", "ingolstadt1", "cologne3"]
MAIN = {"dqn": "DQN", "ddqn": "Double DQN", "ppo": "PPO", "a2c": "A2C"}
LEVELS = ["x0.7", "x1.0", "x1.3", "real"]
FAIL_RATIO = 2.0
RNG = np.random.default_rng(0)


def load(name="test_episodes.csv"):
    e = pd.read_csv(os.path.join(RES, name))
    base = pd.read_csv(os.path.join(RES, "test_episodes.csv"))
    fixed = base[base.controller == "fixed"].set_index(["scenario", "route_file"])["avg_total_wait"]
    e["fail"] = [w > FAIL_RATIO * fixed[(s, f)] for s, f, w in zip(e.scenario, e.route_file, e.avg_total_wait)]
    return e, fixed


MAIN_SEEDS = 5          # main comparison: seeds 0-4 for every learner (cologne3 A2C has extra seeds 5-9)
BASELINES = ["fixed", "actuated", "maxpressure", "random"]


def main_study(e):
    """Drop the extra seeds that were trained only for the ablation."""
    return e[(e.seed < MAIN_SEEDS) | e.controller.isin(BASELINES)]


def label(controller):
    return MAIN.get(controller.split("_")[0], controller) if controller.endswith("_64x64_variants") else controller


def bootstrap_ci(per_seed, n=10000):
    """95% CI of the mean of per-seed means, resampling seeds."""
    v = np.asarray(per_seed, float)
    if len(v) < 2:
        return np.nan, np.nan
    means = RNG.choice(v, size=(n, len(v)), replace=True).mean(axis=1)
    return np.percentile(means, 2.5), np.percentile(means, 97.5)


def wilson(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def cvar(x, alpha=0.10):
    x = np.sort(np.asarray(x, float))[::-1]
    k = max(1, int(np.ceil(alpha * len(x))))
    return x[:k].mean()


def holm(pvals):
    order = np.argsort(pvals)
    adj = np.empty(len(pvals))
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(pvals) - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj


def table_robust(e):
    e = main_study(e)
    rows = []
    for s in SCEN:
        for c in [b for b in BASELINES if b in set(e.controller)] + [f"{a}_64x64_variants" for a in MAIN]:
            d = e[(e.scenario == s) & (e.controller == c)]
            if d.empty:          # e.g. a baseline not yet evaluated for this scenario
                continue
            per_seed = d.groupby("seed").avg_total_wait.mean()
            lo, hi = bootstrap_ci(per_seed)
            k, n = int(d.fail.sum()), len(d)
            flo, fhi = wilson(k, n)
            rows.append(dict(scenario=s, controller=label(c), mean=d.avg_total_wait.mean(), ci_lo=lo, ci_hi=hi,
                             median=d.avg_total_wait.median(), p95=np.percentile(d.avg_total_wait, 95),
                             cvar10=cvar(d.avg_total_wait), fail=k, n=n, fail_p=k / n, fail_lo=flo, fail_hi=fhi))
    return pd.DataFrame(rows)


def table_levels(e):
    e = main_study(e)
    rows = []
    for s in SCEN:
        for c in [b for b in ["fixed", "maxpressure"] if b in set(e.controller)] + [f"{a}_64x64_variants" for a in MAIN]:
            for lv in LEVELS:
                d = e[(e.scenario == s) & (e.controller == c) & (e.demand == lv)]
                if d.empty:
                    continue
                rows.append(dict(scenario=s, controller=label(c), level=lv, median=d.avg_total_wait.median(),
                                 fail=int(d.fail.sum()), n=len(d)))
    return pd.DataFrame(rows)


def paired_tests(e):
    """Wilcoxon signed-rank on the 10 identical test files (per-file mean over seeds)."""
    e = main_study(e)
    rows = []
    for s in SCEN:
        d = e[e.scenario == s]
        per_file = d.groupby(["controller", "route_file"]).avg_total_wait.mean().unstack(0)
        fixed = per_file["fixed"]
        if "maxpressure" in per_file:
            rows.append(dict(scenario=s, comparison="MaxPressure vs fixed-time",
                             median_diff=float(np.median(per_file["maxpressure"] - fixed)),
                             p=stats.wilcoxon(per_file["maxpressure"], fixed).pvalue))
            rows.append(dict(scenario=s, comparison="A2C vs MaxPressure",
                             median_diff=float(np.median(per_file["a2c_64x64_variants"] - per_file["maxpressure"])),
                             p=stats.wilcoxon(per_file["a2c_64x64_variants"], per_file["maxpressure"]).pvalue))
        for a in MAIN:
            c = f"{a}_64x64_variants"
            rows.append(dict(scenario=s, comparison=f"{MAIN[a]} vs fixed-time",
                             median_diff=float(np.median(per_file[c] - fixed)),
                             p=stats.wilcoxon(per_file[c], fixed).pvalue))
        for a in ["dqn", "ddqn", "ppo"]:
            c = f"{a}_64x64_variants"
            rows.append(dict(scenario=s, comparison=f"A2C vs {MAIN[a]}",
                             median_diff=float(np.median(per_file["a2c_64x64_variants"] - per_file[c])),
                             p=stats.wilcoxon(per_file["a2c_64x64_variants"], per_file[c]).pvalue))
    df = pd.DataFrame(rows)
    df["p_holm"] = holm(df.p.values)
    return df


def ablation_tests(e):
    variants = {"full (C1+C2)": "a2c_64x64_variants", "-C1 real hour only": "a2c_64x64_real",
                "-C2 independent": "a2c-ind_64x64_variants", "+ masking": "a2c-mask_64x64_variants"}
    rows = []
    for s in ["cologne1", "cologne3"]:
        full = e[(e.scenario == s) & (e.controller == variants["full (C1+C2)"])]
        full_file = full.groupby("route_file").avg_total_wait.mean()
        f_ep, f_seed = int(full.fail.sum()), int(full.groupby("seed").fail.any().sum())
        f_n, f_ns = len(full), full.seed.nunique()
        for name, c in variants.items():
            d = e[(e.scenario == s) & (e.controller == c)]
            if d.empty:
                continue
            k_ep, k_seed = int(d.fail.sum()), int(d.groupby("seed").fail.any().sum())
            if c == variants["full (C1+C2)"]:
                p_ep = p_seed = p_file = np.nan
            else:
                n, ns = len(d), d.seed.nunique()
                p_ep = stats.fisher_exact([[f_ep, f_n - f_ep], [k_ep, n - k_ep]]).pvalue
                p_seed = stats.fisher_exact([[f_seed, f_ns - f_seed], [k_seed, ns - k_seed]]).pvalue
                p_file = stats.wilcoxon(d.groupby("route_file").avg_total_wait.mean(), full_file).pvalue
            rows.append(dict(scenario=s, variant=name, seeds=d.seed.nunique(), episodes=len(d),
                             failed_episodes=k_ep, failing_seeds=k_seed,
                             fisher_p_episodes=p_ep, fisher_p_seeds=p_seed, wilcoxon_p_files=p_file,
                             median_episode=d.avg_total_wait.median()))
    return pd.DataFrame(rows)


def checkpoint_compare():
    path = os.path.join(RES, "test_final_episodes.csv")
    if not os.path.exists(path) or os.path.getsize(path) < 200:   # missing, or evaluation still starting
        return None
    best, _ = load()
    final, _ = load("test_final_episodes.csv")
    rows = []
    for s in SCEN:
        for a in MAIN:
            c = f"{a}_64x64_variants"
            b = best[(best.scenario == s) & (best.controller == c) & (best.seed < MAIN_SEEDS)]
            f = final[(final.scenario == s) & (final.controller == c)]
            if len(f) < len(b):          # final-model evaluation not complete for this algorithm yet
                continue
            rows.append(dict(scenario=s, algo=MAIN[a], best_mean=b.avg_total_wait.mean(), best_median=b.avg_total_wait.median(),
                             best_fail=int(b.fail.sum()), final_mean=f.avg_total_wait.mean(),
                             final_median=f.avg_total_wait.median(), final_fail=int(f.fail.sum()), n=len(f)))
    return pd.DataFrame(rows)


def dqn_sensitivity(e):
    """Reviewer question 2: DQN with one hyperparameter changed, against default DQN on the same seeds."""
    variants = {"default": "dqn_64x64_variants", "replay buffer 200k": "dqn-buf200k_64x64_variants",
                "target update 2000": "dqn-tgt2000_64x64_variants", "exploration 30%": "dqn-exp30_64x64_variants"}
    rows = []
    for s in SCEN:
        trained = [c for c in variants.values() if c != "dqn_64x64_variants" and ((e.scenario == s) & (e.controller == c)).any()]
        if not trained:
            continue
        seeds = sorted(e[(e.scenario == s) & (e.controller == trained[0])].seed.unique())
        base = e[(e.scenario == s) & (e.controller == "dqn_64x64_variants") & e.seed.isin(seeds)]
        for name, c in variants.items():
            d = e[(e.scenario == s) & (e.controller == c) & e.seed.isin(seeds)]
            if d.empty:
                continue
            p = np.nan if c == "dqn_64x64_variants" else stats.wilcoxon(
                d.groupby("route_file").avg_total_wait.mean(), base.groupby("route_file").avg_total_wait.mean()).pvalue
            rows.append(dict(scenario=s, variant=name, seeds=len(seeds), mean=d.avg_total_wait.mean(),
                             median=d.avg_total_wait.median(), cvar10=cvar(d.avg_total_wait),
                             failed=int(d.fail.sum()), episodes=len(d), wilcoxon_p_vs_default=p))
    return pd.DataFrame(rows)


def main():
    pd.set_option("display.width", 220)
    e, _ = load()
    out = os.path.join(RES, "robust_stats")
    os.makedirs(out, exist_ok=True)
    parts = {"robust_summary": table_robust(e), "by_level": table_levels(e), "paired_tests": paired_tests(e),
             "ablation_tests": ablation_tests(e), "dqn_sensitivity": dqn_sensitivity(e)}
    ck = checkpoint_compare()
    if ck is not None:
        parts["checkpoint_selection"] = ck
    for name, df in parts.items():
        df.to_csv(os.path.join(out, f"{name}.csv"), index=False)
        print(f"\n== {name}")
        print(df.round(3).to_string(index=False))
    print(f"\nwritten: {out}")


if __name__ == "__main__":
    main()
