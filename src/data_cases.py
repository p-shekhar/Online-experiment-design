from __future__ import annotations

from dataclasses import replace
import tarfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from simulators import SimulationConfig, run_design_simulation, summarize_simulation


def load_criteo_sample(path: Path, nrows: int = 300_000) -> pd.DataFrame:
    cols = [f"f{i}" for i in range(12)] + ["treatment", "conversion", "visit", "exposure"]
    target_per_arm = max(1, nrows // 2)
    treated_parts: list[pd.DataFrame] = []
    control_parts: list[pd.DataFrame] = []
    treated_n = 0
    control_n = 0
    for chunk in pd.read_csv(path, usecols=cols, chunksize=500_000):
        if treated_n < target_per_arm:
            treated = chunk[chunk["treatment"] == 1]
            if not treated.empty:
                treated_parts.append(treated)
                treated_n += len(treated)
        if control_n < target_per_arm:
            control = chunk[chunk["treatment"] == 0]
            if not control.empty:
                control_parts.append(control)
                control_n += len(control)
        if treated_n >= target_per_arm and control_n >= target_per_arm:
            break
    if not treated_parts or not control_parts:
        raise ValueError("Could not read both treatment arms from the Criteo uplift file.")
    treated_df = pd.concat(treated_parts, ignore_index=True).head(target_per_arm)
    control_df = pd.concat(control_parts, ignore_index=True).head(target_per_arm)
    combined = pd.concat([treated_df, control_df], ignore_index=True)
    return combined.sample(frac=1.0, random_state=2026)


def criteo_calibration(sample: pd.DataFrame) -> dict[str, float]:
    treated = sample[sample["treatment"] == 1]
    control = sample[sample["treatment"] == 0]
    visit_lift = treated["visit"].mean() - control["visit"].mean()
    conversion_lift = treated["conversion"].mean() - control["conversion"].mean()
    exposure_rate = sample["exposure"].mean()
    return {
        "n_rows": float(len(sample)),
        "treatment_rate": float(sample["treatment"].mean()),
        "visit_rate": float(sample["visit"].mean()),
        "conversion_rate": float(sample["conversion"].mean()),
        "exposure_rate": float(exposure_rate),
        "visit_lift": float(visit_lift),
        "conversion_lift": float(conversion_lift),
    }


def _first_present(columns: set[str], candidates: list[str]) -> str | None:
    return next((col for col in candidates if col in columns), None)


def _hash_codes(frame: pd.DataFrame, columns: list[str], modulo: int, seed: int) -> np.ndarray:
    if columns:
        hashed = pd.util.hash_pandas_object(frame[columns].astype(str), index=False).to_numpy()
    else:
        hashed = np.arange(len(frame), dtype=np.uint64) + np.uint64(seed)
    return (hashed % max(1, modulo)).astype(int)


def _period_codes(source: pd.DataFrame, time_col: str | None, n_periods: int) -> np.ndarray:
    if time_col is None:
        return (np.arange(len(source)) % n_periods).astype(int)
    values = pd.to_numeric(source[time_col], errors="coerce")
    if values.notna().sum() == 0:
        values = pd.Series(pd.factorize(source[time_col].astype(str))[0], index=source.index)
    values = values.fillna(values.median())
    ranks = values.rank(method="first").to_numpy()
    scaled = np.floor((ranks - 1) / max(len(source), 1) * n_periods)
    return np.clip(scaled, 0, n_periods - 1).astype(int)


def _bounded_int(value: float | int | None, lower: int, upper: int, fallback: int) -> int:
    if value is None or not np.isfinite(float(value)):
        return fallback
    return int(np.clip(round(float(value)), lower, upper))


def _sqrt_count(value: float | int | None, lower: int, upper: int, fallback: int) -> int:
    if value is None or not np.isfinite(float(value)) or float(value) <= 0:
        return fallback
    return _bounded_int(np.sqrt(float(value)), lower, upper, fallback)


def _log_seeded_panel(
    sample: pd.DataFrame,
    config: SimulationConfig,
    outcome_candidates: list[str],
    user_candidates: list[str],
    graph_candidates: list[str],
    budget_candidates: list[str],
    time_candidates: list[str],
) -> pd.DataFrame:
    target_rows = config.n_users * config.n_periods
    source = sample.sample(
        n=target_rows,
        replace=len(sample) < target_rows,
        random_state=config.seed,
    ).reset_index(drop=True)
    columns = set(source.columns)
    outcome_col = _first_present(columns, outcome_candidates)
    time_col = _first_present(columns, time_candidates)
    user_cols = [col for col in user_candidates if col in columns]
    graph_cols = [col for col in graph_candidates if col in columns]
    budget_cols = [col for col in budget_candidates if col in columns]

    if outcome_col is not None:
        outcome = pd.to_numeric(source[outcome_col], errors="coerce")
        outcome = outcome.fillna(outcome.mean())
        centered = outcome - outcome.mean()
        baseline = 0.5 + centered / max(float(outcome.std(ddof=0)), 1.0)
    else:
        rng = np.random.default_rng(config.seed)
        baseline = pd.Series(rng.normal(0.5, 0.20, size=len(source)))

    return pd.DataFrame(
        {
            "user_id": _hash_codes(source, user_cols, config.n_users, config.seed),
            "cluster_id": _hash_codes(source, graph_cols, config.n_clusters, config.seed + 11),
            "budget_group": _hash_codes(
                source,
                budget_cols,
                max(8, config.n_clusters // 2),
                config.seed + 23,
            ),
            "period": _period_codes(source, time_col, config.n_periods),
            "baseline": baseline.to_numpy(dtype=float),
        }
    )


def _paper_ambiguity_grid(domain: str) -> list[dict[str, float | str]]:
    """Full finite theta grid used by the robust selector in the paper."""
    gamma_g_values = [0.0, 0.1, 0.3]
    gamma_b_values = [0.0, 0.2, 0.5]
    lambda_values = [0.0, 0.05, 0.2]
    localities = ["campaign", "advertiser", "exchange-region"]
    mechanisms: list[dict[str, float | str]] = []
    idx = 0
    for gamma_g in gamma_g_values:
        for gamma_b in gamma_b_values:
            for carryover_lambda in lambda_values:
                for locality in localities:
                    mechanisms.append(
                        {
                            "theta_id": f"{domain}_theta_{idx:02d}",
                            "mechanism_label": (
                                f"g{gamma_g:.2f}_b{gamma_b:.2f}_l"
                                f"{carryover_lambda:.2f}_{locality}"
                            ),
                            "gamma_g": gamma_g,
                            "gamma_b": gamma_b,
                            "lambda": carryover_lambda,
                            "locality": locality,
                        }
                    )
                    idx += 1
    return mechanisms


def _locality_clusters(base_clusters: int, locality: str) -> int:
    multiplier = {
        "campaign": 1.35,
        "advertiser": 1.00,
        "exchange-region": 0.65,
    }[locality]
    return max(8, int(round(base_clusters * multiplier)))


SUPPORT_STRESS_BY_DESIGN = {
    "user_randomization": 0.45,
    "cluster_randomization": 0.22,
    "switchback": 0.12,
    "budget_split": 0.25,
    "two_stage_saturation": 0.28,
    "mixed_randomization": 0.30,
}


def _add_support_stress(summary: pd.DataFrame, support_risk: float) -> pd.DataFrame:
    """Add a direct known-propensity support stress to risk components.

    Open Bandit exposes logged action probabilities. When the inverse-propensity
    effective sample share is small, the design-selection problem should carry
    a support penalty instead of treating the logged bandit data as equally
    informative. The penalty is design-sensitive and enters the same empirical
    components discussed in the paper: contamination/carryover and residual
    estimand mismatch.
    """
    df = summary.copy()
    support_risk = float(np.clip(support_risk, 0.0, 1.0))
    df["support_risk"] = support_risk
    sensitivity = df["design"].map(SUPPORT_STRESS_BY_DESIGN).fillna(0.30).astype(float)
    df["support_stress_adjustment"] = support_risk * sensitivity
    if support_risk > 0:
        df["contamination_risk"] = df["contamination_risk"] + df["support_stress_adjustment"]
        df["estimand_mismatch"] = df["estimand_mismatch"] + 0.50 * df["support_stress_adjustment"]
    return df


def _run_ambiguity_case(
    base_config: SimulationConfig,
    domain: str,
    case: str,
    graph_share: float,
    n_reps: int,
    metadata: dict[str, float | str | bool],
    base_panel: pd.DataFrame | None = None,
) -> pd.DataFrame:
    mechanisms = _paper_ambiguity_grid(domain)
    reps_per_mechanism = max(2, int(np.ceil(n_reps / len(mechanisms))))
    graph_share = float(np.clip(graph_share, 0.0, 1.0))
    budget_share = 1.0 - graph_share
    rows = []
    for idx, mechanism in enumerate(mechanisms):
        gamma_g = float(mechanism["gamma_g"])
        gamma_b = float(mechanism["gamma_b"])
        carry = float(mechanism["lambda"])
        graph_component = base_config.spillover_strength * graph_share * (gamma_g / 0.3)
        budget_component = base_config.spillover_strength * budget_share * (gamma_b / 0.5)
        config = replace(
            base_config,
            n_clusters=_locality_clusters(base_config.n_clusters, str(mechanism["locality"])),
            spillover_strength=graph_component + budget_component,
            graph_spillover_strength=graph_component,
            budget_spillover_strength=budget_component,
            carryover_strength=base_config.carryover_strength * (carry / 0.2),
            gamma_g=gamma_g,
            gamma_b=gamma_b,
            carryover_lambda=carry,
            locality=str(mechanism["locality"]),
            seed=base_config.seed + 9973 * idx,
        )
        summary = summarize_simulation(
            run_design_simulation(config, n_reps=reps_per_mechanism, base_panel=base_panel)
        )
        summary["case"] = case
        summary["ambiguity_set_size"] = len(mechanisms)
        summary["reps_per_mechanism"] = reps_per_mechanism
        summary["sim_n_users"] = config.n_users
        summary["sim_n_clusters"] = config.n_clusters
        summary["sim_n_periods"] = config.n_periods
        summary["calibration_graph_share"] = graph_share
        summary["calibration_budget_share"] = budget_share
        summary = _add_support_stress(
            summary,
            float(metadata.get("support_risk", 0.0) or 0.0),
        )
        for key, value in mechanism.items():
            summary[key] = value
        for key, value in metadata.items():
            summary[key] = value
        rows.append(summary)
    return pd.concat(rows, ignore_index=True)


def run_criteo_ads_design_case(sample: pd.DataFrame, n_reps: int = 120) -> pd.DataFrame:
    stats = criteo_calibration(sample)
    user_feature_units = float(sample[["f0", "f1", "f2"]].drop_duplicates().shape[0])
    graph_feature_units = float(sample[["f3", "f4", "f5"]].drop_duplicates().shape[0])
    budget_feature_units = float(sample[["f6", "f7", "f8"]].drop_duplicates().shape[0])
    exposure_rate = max(float(stats["exposure_rate"]), 1e-6)
    direct = max(abs(stats["visit_lift"]), 0.015)
    triggered_multiplier = 1.0 + min(0.50, 6.0 * exposure_rate)
    budget_share = min(0.90, max(0.65, 1.0 - 7.5 * exposure_rate))
    graph_share = 1.0 - budget_share
    base_config = SimulationConfig(
        n_users=3000,
        n_clusters=60,
        n_periods=14,
        direct_effect=direct,
        spillover_strength=max(0.8 * direct * triggered_multiplier, 0.012),
        carryover_strength=max(0.25 * direct * (1.0 + min(0.25, 3.0 * exposure_rate)), 0.004),
        noise_sd=0.22,
        seed=2026,
    )
    return _run_ambiguity_case(
        base_config=base_config,
        domain="criteo_ads",
        case="criteo_ads_budget_interference",
        graph_share=graph_share,
        n_reps=n_reps,
        base_panel=_log_seeded_panel(
            sample,
            base_config,
            outcome_candidates=["visit", "conversion"],
            user_candidates=["f0", "f1", "f2"],
            graph_candidates=["f3", "f4", "f5"],
            budget_candidates=["f6", "f7", "f8"],
            time_candidates=[],
        ),
        metadata={
            "calibration_visit_lift": stats["visit_lift"],
            "calibration_conversion_lift": stats["conversion_lift"],
            "calibration_exposure_rate": stats["exposure_rate"],
            "calibration_direct_effect": direct,
            "calibration_triggered_multiplier": triggered_multiplier,
            "calibration_spillover_strength": base_config.spillover_strength,
            "calibration_carryover_strength": base_config.carryover_strength,
            "calibration_feature_user_units": user_feature_units,
            "calibration_feature_graph_units": graph_feature_units,
            "calibration_feature_budget_units": budget_feature_units,
        },
    )


def load_open_bandit_sample(
    zip_path: Path,
    behavior_policy: str = "random",
    campaign: str = "men",
    nrows: int | None = 300_000,
) -> pd.DataFrame:
    member = f"open_bandit_dataset/{behavior_policy}/{campaign}/{campaign}.csv"
    usecols = [
        "timestamp",
        "item_id",
        "position",
        "click",
        "propensity_score",
        "user_feature_0",
        "user_feature_1",
        "user_feature_2",
        "user_feature_3",
    ]
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open(member) as handle:
            return pd.read_csv(handle, usecols=usecols, nrows=nrows)


def open_bandit_calibration(sample: pd.DataFrame) -> dict[str, float]:
    ps = sample["propensity_score"].clip(lower=1e-6)
    weights = 1.0 / ps
    ess = weights.sum() ** 2 / (weights**2).sum()
    return {
        "n_rows": float(len(sample)),
        "n_items": float(sample["item_id"].nunique()),
        "n_positions": float(sample["position"].nunique()),
        "click_rate": float(sample["click"].mean()),
        "mean_propensity": float(ps.mean()),
        "min_propensity": float(ps.min()),
        "max_propensity": float(ps.max()),
        "ips_effective_sample_size": float(ess),
        "ips_ess_share": float(ess / len(sample)),
    }


def open_bandit_propensity_diagnostics(sample: pd.DataFrame) -> pd.DataFrame:
    rows = []
    ps = sample["propensity_score"].clip(lower=1e-6)
    reward = sample["click"].astype(float)
    raw_weights = 1.0 / ps
    for cap in [np.inf, 100.0, 50.0, 25.0, 10.0]:
        weights = raw_weights.clip(upper=cap)
        ess = weights.sum() ** 2 / (weights**2).sum()
        rows.append(
            {
                "clip_cap": "none" if np.isinf(cap) else cap,
                "weighted_click_rate": float((weights * reward).sum() / weights.sum()),
                "effective_sample_size": float(ess),
                "ess_share": float(ess / len(sample)),
                "weight_mean": float(weights.mean()),
                "weight_sd": float(weights.std(ddof=1)),
                "weight_max": float(weights.max()),
            }
        )
    return pd.DataFrame(rows)


def run_open_bandit_known_propensity_case(
    sample: pd.DataFrame,
    n_reps: int = 120,
) -> pd.DataFrame:
    stats = open_bandit_calibration(sample)
    direct = max(0.015, 2.5 * stats["click_rate"])
    support_risk = min(1.0, max(0.0, 1.0 - stats["ips_ess_share"]))
    propensity_noise = min(0.08, max(0.01, support_risk))
    n_clusters = _bounded_int(np.sqrt(stats["n_items"]) * 12, 40, 120, 60)
    base_config = SimulationConfig(
        n_users=2600,
        n_clusters=n_clusters,
        n_periods=14,
        direct_effect=direct,
        spillover_strength=0.45 * direct + 0.04 * propensity_noise,
        carryover_strength=0.20 * direct,
        noise_sd=0.20,
        seed=2525,
    )
    return _run_ambiguity_case(
        base_config=base_config,
        domain="open_bandit",
        case="open_bandit_known_propensity_recommendation",
        graph_share=0.70,
        n_reps=n_reps,
        base_panel=_log_seeded_panel(
            sample,
            base_config,
            outcome_candidates=["click"],
            user_candidates=[
                "user_feature_0",
                "user_feature_1",
                "user_feature_2",
                "user_feature_3",
            ],
            graph_candidates=["item_id", "position"],
            budget_candidates=["item_id"],
            time_candidates=["timestamp"],
        ),
        metadata={
            "calibration_click_rate": stats["click_rate"],
            "calibration_ips_ess_share": stats["ips_ess_share"],
            "calibration_mean_propensity": stats["mean_propensity"],
            "calibration_min_propensity": stats["min_propensity"],
            "calibration_max_propensity": stats["max_propensity"],
            "calibration_direct_effect": direct,
            "calibration_spillover_strength": base_config.spillover_strength,
            "calibration_carryover_strength": base_config.carryover_strength,
            "support_risk": support_risk,
        },
    )


def kuairand_pure_member_table(zip_path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open("KuaiRand-Pure.tar.gz") as nested:
            with tarfile.open(fileobj=nested, mode="r:gz") as tf:
                rows = [
                    {"member": m.name, "size_mb": round(m.size / (1024**2), 3)}
                    for m in tf.getmembers()
                    if m.isfile()
                ]
    return pd.DataFrame(rows)


def load_kuairand_pure_sample(zip_path: Path, nrows: int = 250_000) -> pd.DataFrame:
    """Read the first interaction-like CSV from KuaiRand-Pure without unpacking the full archive."""
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open("KuaiRand-Pure.tar.gz") as nested:
            with tarfile.open(fileobj=nested, mode="r:gz") as tf:
                candidates = [
                    m
                    for m in tf.getmembers()
                    if m.isfile() and m.name.lower().endswith(".csv") and "log" in m.name.lower()
                ]
                if not candidates:
                    candidates = [
                        m for m in tf.getmembers() if m.isfile() and m.name.lower().endswith(".csv")
                    ]
                if not candidates:
                    raise FileNotFoundError("No CSV file found inside KuaiRand-Pure.tar.gz.")
                member = sorted(candidates, key=lambda m: m.size, reverse=True)[0]
                handle = tf.extractfile(member)
                if handle is None:
                    raise FileNotFoundError(f"Could not open {member.name}.")
                return pd.read_csv(handle, nrows=nrows)


def kuairand_calibration(sample: pd.DataFrame) -> dict[str, float]:
    cols = set(sample.columns)
    user_col = next((c for c in ["user_id", "user", "uid"] if c in cols), None)
    item_col = next((c for c in ["video_id", "item_id", "photo_id", "vid"] if c in cols), None)
    time_col = next((c for c in ["time_ms", "timestamp", "date", "time"] if c in cols), None)
    outcome_col = next(
        (
            c
            for c in ["is_click", "click", "is_like", "like", "long_view", "play_time_ms"]
            if c in cols
        ),
        None,
    )
    if outcome_col is None:
        numeric_cols = sample.select_dtypes(include=[np.number]).columns
        outcome_col = numeric_cols[-1] if len(numeric_cols) else None
    outcome_mean = float(sample[outcome_col].mean()) if outcome_col else np.nan
    return {
        "n_rows": float(len(sample)),
        "n_users": float(sample[user_col].nunique()) if user_col else np.nan,
        "n_items": float(sample[item_col].nunique()) if item_col else np.nan,
        "has_time": bool(time_col),
        "outcome_col": outcome_col or "",
        "outcome_mean": outcome_mean,
    }


def run_kuairand_carryover_case(sample: pd.DataFrame, n_reps: int = 120) -> pd.DataFrame:
    stats = kuairand_calibration(sample)
    base_effect = max(0.02, min(0.10, abs(stats.get("outcome_mean", 0.05)) * 0.25))
    n_users = _bounded_int(stats.get("n_users"), 1500, 3500, 2500)
    n_clusters = _sqrt_count(stats.get("n_items"), 50, 120, 75)
    n_periods = 21 if bool(stats.get("has_time", False)) else 14
    base_config = SimulationConfig(
        n_users=n_users,
        n_clusters=n_clusters,
        n_periods=n_periods,
        direct_effect=base_effect,
        spillover_strength=0.75 * base_effect,
        carryover_strength=0.65 * base_effect,
        noise_sd=0.24,
        seed=3031,
    )
    return _run_ambiguity_case(
        base_config=base_config,
        domain="kuairand",
        case="kuairand_member_experience_carryover",
        graph_share=0.70,
        n_reps=n_reps,
        base_panel=_log_seeded_panel(
            sample,
            base_config,
            outcome_candidates=[
                "is_click",
                "click",
                "is_like",
                "like",
                "long_view",
                "play_time_ms",
            ],
            user_candidates=["user_id", "user", "uid"],
            graph_candidates=["video_id", "item_id", "photo_id", "vid"],
            budget_candidates=["video_id", "item_id", "photo_id", "vid"],
            time_candidates=["time_ms", "timestamp", "date", "time"],
        ),
        metadata={
            "calibration_outcome_col": stats["outcome_col"],
            "calibration_outcome_mean": stats["outcome_mean"],
            "calibration_n_users": stats["n_users"],
            "calibration_n_items": stats["n_items"],
            "calibration_has_time": stats["has_time"],
            "calibration_direct_effect": base_effect,
            "calibration_spillover_strength": base_config.spillover_strength,
            "calibration_carryover_strength": base_config.carryover_strength,
        },
    )


def load_movielens_sample(path: Path, nrows: int = 500_000) -> pd.DataFrame:
    with zipfile.ZipFile(path) as zf:
        with zf.open("ml-32m/ratings.csv") as handle:
            return pd.read_csv(handle, nrows=nrows)


def movielens_calibration(sample: pd.DataFrame) -> dict[str, float]:
    return {
        "n_rows": float(len(sample)),
        "n_users": float(sample["userId"].nunique()),
        "n_items": float(sample["movieId"].nunique()),
        "mean_rating": float(sample["rating"].mean()),
        "rating_sd": float(sample["rating"].std(ddof=1)),
    }


def run_movielens_graph_case(sample: pd.DataFrame, n_reps: int = 120) -> pd.DataFrame:
    stats = movielens_calibration(sample)
    direct = 0.08 * stats["rating_sd"]
    n_users = _bounded_int(stats.get("n_users"), 2000, 4000, 3200)
    n_clusters = _sqrt_count(stats.get("n_items"), 60, 140, 80)
    base_config = SimulationConfig(
        n_users=n_users,
        n_clusters=n_clusters,
        n_periods=10,
        direct_effect=direct,
        spillover_strength=1.25 * direct,
        carryover_strength=0.15 * direct,
        noise_sd=0.35 * stats["rating_sd"],
        seed=4042,
    )
    return _run_ambiguity_case(
        base_config=base_config,
        domain="movielens",
        case="movielens_graph_interference",
        graph_share=0.90,
        n_reps=n_reps,
        base_panel=_log_seeded_panel(
            sample,
            base_config,
            outcome_candidates=["rating"],
            user_candidates=["userId"],
            graph_candidates=["movieId"],
            budget_candidates=["movieId"],
            time_candidates=["timestamp"],
        ),
        metadata={
            "calibration_rating_sd": stats["rating_sd"],
            "calibration_n_users": stats["n_users"],
            "calibration_n_items": stats["n_items"],
            "calibration_direct_effect": direct,
            "calibration_spillover_strength": base_config.spillover_strength,
            "calibration_carryover_strength": base_config.carryover_strength,
        },
    )
