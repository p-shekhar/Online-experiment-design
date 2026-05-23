from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from designs import design_catalog_frame, mde_from_assignment_units


@dataclass(frozen=True)
class SimulationConfig:
    n_users: int = 2500
    n_clusters: int = 50
    n_periods: int = 14
    direct_effect: float = 0.08
    spillover_strength: float = 0.06
    graph_spillover_strength: float | None = None
    budget_spillover_strength: float | None = None
    carryover_strength: float = 0.03
    gamma_g: float = 1.0
    gamma_b: float = 1.0
    carryover_lambda: float = 1.0
    locality: str = "advertiser"
    noise_sd: float = 0.25
    seed: int = 123


def _graph_strength(config: SimulationConfig) -> float:
    if config.graph_spillover_strength is not None:
        return config.graph_spillover_strength
    return 0.65 * config.spillover_strength


def _budget_strength(config: SimulationConfig) -> float:
    if config.budget_spillover_strength is not None:
        return config.budget_spillover_strength
    return 0.35 * config.spillover_strength


def make_platform_panel(config: SimulationConfig) -> pd.DataFrame:
    rng = np.random.default_rng(config.seed)
    users = np.arange(config.n_users)
    clusters = rng.integers(0, config.n_clusters, size=config.n_users)
    budget_groups = rng.integers(0, max(8, config.n_clusters // 2), size=config.n_users)
    user_quality = rng.normal(0, 0.25, size=config.n_users)
    rows = []
    for period in range(config.n_periods):
        seasonal = 0.08 * np.sin(2 * np.pi * period / 7)
        base = 0.5 + user_quality + seasonal + rng.normal(0, 0.03, size=config.n_users)
        rows.append(
            pd.DataFrame(
                {
                    "user_id": users,
                    "cluster_id": clusters,
                    "budget_group": budget_groups,
                    "period": period,
                    "baseline": base,
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def assign_treatment(panel: pd.DataFrame, design: str, rng: np.random.Generator) -> pd.Series:
    df = panel
    if design == "user_randomization":
        user_ids = np.sort(df["user_id"].unique())
        user_z = pd.Series(rng.binomial(1, 0.5, len(user_ids)), index=user_ids)
        return df["user_id"].map(user_z).astype(int)
    if design == "cluster_randomization":
        cluster_ids = np.sort(df["cluster_id"].unique())
        cluster_z = pd.Series(
            rng.binomial(1, 0.5, len(cluster_ids)),
            index=cluster_ids,
        )
        return df["cluster_id"].map(cluster_z).astype(int)
    if design == "switchback":
        periods = np.sort(df["period"].unique())
        period_z = pd.Series(rng.binomial(1, 0.5, len(periods)), index=periods)
        return df["period"].map(period_z).astype(int)
    if design == "budget_split":
        budget_groups = np.sort(df["budget_group"].unique())
        budget_z = pd.Series(
            rng.binomial(1, 0.5, len(budget_groups)),
            index=budget_groups,
        )
        return df["budget_group"].map(budget_z).astype(int)
    if design == "two_stage_saturation":
        clusters = np.sort(df["cluster_id"].unique())
        saturation = pd.Series(rng.choice([0.25, 0.50, 0.75], size=len(clusters)), index=clusters)
        probs = df["cluster_id"].map(saturation).to_numpy()
        return pd.Series(rng.binomial(1, probs), index=df.index)
    if design == "mixed_randomization":
        keys = df[["cluster_id", "period"]].drop_duplicates().sort_values(["cluster_id", "period"])
        key_z = pd.Series(rng.binomial(1, 0.5, len(keys)), index=list(map(tuple, keys.to_numpy())))
        row_keys = pd.Series(
            list(map(tuple, df[["cluster_id", "period"]].to_numpy())),
            index=df.index,
        )
        return row_keys.map(key_z).astype(int)
    raise ValueError(f"Unknown design: {design}")


def _assignment_unit(panel: pd.DataFrame, design: str) -> pd.Series:
    if design == "user_randomization":
        return panel["user_id"].astype(str)
    if design == "cluster_randomization":
        return panel["cluster_id"].astype(str)
    if design == "switchback":
        return panel["period"].astype(str)
    if design == "budget_split":
        return panel["budget_group"].astype(str)
    if design == "two_stage_saturation":
        return panel["cluster_id"].astype(str) + "_" + panel["period"].astype(str)
    if design == "mixed_randomization":
        return panel["cluster_id"].astype(str) + "_" + panel["period"].astype(str)
    raise ValueError(f"Unknown design: {design}")


def observe_outcomes(
    panel: pd.DataFrame,
    z: pd.Series,
    config: SimulationConfig,
    rng: np.random.Generator,
) -> pd.DataFrame:
    df = panel.copy()
    df["z"] = z.to_numpy()
    cluster_share = df.groupby(["cluster_id", "period"])["z"].transform("mean")
    budget_share = df.groupby(["budget_group", "period"])["z"].transform("mean")
    df["graph_exposure"] = cluster_share
    df["budget_exposure"] = budget_share
    df["spillover_exposure"] = (
        config.gamma_g * df["graph_exposure"] + config.gamma_b * df["budget_exposure"]
    )
    exposure_scale = max(config.gamma_g + config.gamma_b, 1e-12)
    df["spillover_exposure"] = df["spillover_exposure"] / exposure_scale
    df["lag_z"] = df.sort_values(["user_id", "period"]).groupby("user_id")["z"].shift(1).fillna(0.0)
    df["theta_gamma_g"] = config.gamma_g
    df["theta_gamma_b"] = config.gamma_b
    df["theta_lambda"] = config.carryover_lambda
    df["theta_locality"] = config.locality
    noise = rng.normal(0, config.noise_sd, len(df))
    df["outcome"] = (
        df["baseline"]
        + config.direct_effect * df["z"]
        + _graph_strength(config) * df["graph_exposure"]
        + _budget_strength(config) * df["budget_exposure"]
        + config.carryover_strength * df["lag_z"]
        + noise
    )
    return df


def estimate_design(df: pd.DataFrame, design: str) -> dict[str, float | str]:
    unit = _assignment_unit(df, design)
    unit_df = (
        df.assign(assignment_unit=unit)
        .groupby("assignment_unit", as_index=False)
        .agg(outcome=("outcome", "mean"), z=("z", "mean"))
    )
    treated = unit_df[unit_df["z"] >= 0.5]["outcome"]
    control = unit_df[unit_df["z"] < 0.5]["outcome"]
    estimate = treated.mean() - control.mean()
    unit_sd = unit_df["outcome"].std(ddof=1)
    gamma_g = float(df["theta_gamma_g"].iloc[0]) if "theta_gamma_g" in df else 1.0
    gamma_b = float(df["theta_gamma_b"].iloc[0]) if "theta_gamma_b" in df else 1.0
    carryover_lambda = float(df["theta_lambda"].iloc[0]) if "theta_lambda" in df else 1.0
    exposure_scale = max(1.0 + gamma_g + gamma_b + carryover_lambda, 1e-12)
    # Empirical W1 proxy in the paper's normalized exposure-feature space:
    # (Z_i, gamma_b budget exposure, gamma_g graph exposure, lambda lag exposure).
    # Full launch has direct treatment, full graph/budget exposure, and steady-state lagged exposure.
    exposure_distance = (
        (1.0 - df["z"]).abs()
        + gamma_b * (1.0 - df["budget_exposure"]).abs()
        + gamma_g * (1.0 - df["graph_exposure"]).abs()
        + carryover_lambda * (1.0 - df["lag_z"]).abs()
    ).mean() / exposure_scale
    control_graph_exposure = ((1.0 - df["z"]) * df["graph_exposure"]).mean()
    control_budget_exposure = ((1.0 - df["z"]) * df["budget_exposure"]).mean()
    lag_switching = (df["z"] - df["lag_z"]).abs().mean()
    contamination_scale = max(gamma_g + gamma_b + carryover_lambda, 1e-12)
    contamination_risk = (
        gamma_g * control_graph_exposure
        + gamma_b * control_budget_exposure
        + carryover_lambda * lag_switching
    ) / contamination_scale
    estimand_mismatch = (
        abs(1.0 - df["z"].mean())
        + gamma_b * abs(1.0 - df["budget_exposure"].mean())
        + gamma_g * abs(1.0 - df["graph_exposure"].mean())
        + carryover_lambda * abs(1.0 - df["lag_z"].mean())
    ) / exposure_scale
    return {
        "design": design,
        "estimate": float(estimate),
        "assignment_units": int(len(unit_df)),
        "assignment_unit_sd": float(unit_sd),
        "assignment_unit_variance": float(unit_sd**2),
        "exposure_distance": float(exposure_distance),
        "spillover_contamination": float(
            (
                gamma_g * control_graph_exposure
                + gamma_b * control_budget_exposure
            )
            / max(gamma_g + gamma_b, 1e-12)
        ),
        "carryover_contamination": float(lag_switching),
        "contamination_risk": float(contamination_risk),
        "estimand_mismatch": float(estimand_mismatch),
        "planning_mde": float(mde_from_assignment_units(float(unit_sd), int(len(unit_df)))),
    }


def run_design_simulation(
    config: SimulationConfig,
    designs: list[str] | None = None,
    n_reps: int = 150,
    base_panel: pd.DataFrame | None = None,
) -> pd.DataFrame:
    designs = designs or design_catalog_frame()["name"].tolist()
    if base_panel is None:
        base_panel = make_platform_panel(config)
    else:
        base_panel = base_panel.copy()
    true_launch_effect = (
        config.direct_effect
        + _graph_strength(config)
        + _budget_strength(config)
        + config.carryover_strength
    )
    rows = []
    for rep in range(n_reps):
        for design in designs:
            stable_design_offset = sum(ord(char) for char in design) % 997
            rng = np.random.default_rng(config.seed + 1009 * rep + stable_design_offset)
            z = assign_treatment(base_panel, design, rng)
            observed = observe_outcomes(base_panel, z, config, rng)
            rows.append(
                {
                    "rep": rep,
                    "true_launch_effect": true_launch_effect,
                    **estimate_design(observed, design),
                }
            )
    return pd.DataFrame(rows)


def summarize_simulation(results: pd.DataFrame) -> pd.DataFrame:
    catalog = design_catalog_frame()[["name", "operational_cost"]].rename(
        columns={"name": "design"}
    )
    summary = (
        results.groupby("design")
            .agg(
                mean_estimate=("estimate", "mean"),
                sd=("estimate", "std"),
                mean_mde=("planning_mde", "mean"),
                assignment_units=("assignment_units", "mean"),
                exposure_distance=("exposure_distance", "mean"),
                assignment_unit_variance=("assignment_unit_variance", "mean"),
                planning_mde=("planning_mde", "mean"),
                spillover_contamination=("spillover_contamination", "mean"),
                carryover_contamination=("carryover_contamination", "mean"),
                contamination_risk=("contamination_risk", "mean"),
                estimand_mismatch=("estimand_mismatch", "mean"),
                true_launch_effect=("true_launch_effect", "first"),
            )
            .reset_index()
    )
    summary["bias"] = summary["mean_estimate"] - summary["true_launch_effect"]
    summary["abs_bias"] = summary["bias"].abs()
    summary["mde"] = summary["planning_mde"]
    return summary.merge(catalog, on="design", how="left")
