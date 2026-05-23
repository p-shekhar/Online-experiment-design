from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ExperimentDesign:
    name: str
    assignment_unit: str
    estimand: str
    interference_coverage: str
    operational_cost: float
    estimand_mismatch: float
    notes: str


def design_catalog() -> list[ExperimentDesign]:
    return [
        ExperimentDesign(
            "user_randomization",
            "user",
            "direct user-level intent-to-treat effect",
            "low",
            0.10,
            0.65,
            "High power but vulnerable to shared inventory, budget, producer, "
            "and network spillovers.",
        ),
        ExperimentDesign(
            "cluster_randomization",
            "cluster",
            "cluster-level total effect under local interference",
            "medium",
            0.35,
            0.30,
            "Useful when a graph or market partition captures most spillovers.",
        ),
        ExperimentDesign(
            "switchback",
            "time block",
            "time-local marketplace or product-surface effect",
            "medium-high",
            0.45,
            0.25,
            "Useful when all users share fast-moving state; requires carryover control.",
        ),
        ExperimentDesign(
            "budget_split",
            "budget universe",
            "budget-isolated marketplace effect",
            "high for budget/pacing",
            0.70,
            0.20,
            "Credible for ads budget interference but operationally expensive.",
        ),
        ExperimentDesign(
            "two_stage_saturation",
            "cluster and saturation cell",
            "direct and spillover decomposition",
            "high for measurable spillovers",
            0.80,
            0.15,
            "Best for learning spillover response curves; usually lower throughput.",
        ),
        ExperimentDesign(
            "mixed_randomization",
            "multiple axes",
            "multi-population effect with selected spillover contrasts",
            "high",
            0.90,
            0.10,
            "Flexible but complex; useful when multiple unit types interact.",
        ),
    ]


def design_catalog_frame() -> pd.DataFrame:
    return pd.DataFrame([d.__dict__ for d in design_catalog()])


def mde_from_assignment_units(
    assignment_unit_sd: float,
    n_effective_units: int,
    alpha: float = 0.05,
    power: float = 0.80,
) -> float:
    """Two-arm normal-approximation MDE on the original outcome scale."""
    if n_effective_units <= 1:
        return np.inf
    z_alpha = 1.96 if alpha == 0.05 else 1.96
    z_power = 0.84 if power == 0.80 else 0.84
    return (z_alpha + z_power) * np.sqrt(2 * assignment_unit_sd**2 / n_effective_units)


DEFAULT_RISK_WEIGHTS = {
    "exposure_distance": 1.00,
    "assignment_unit_variance": 0.80,
    "planning_mde": 0.75,
    "contamination_risk": 0.45,
    "operational_cost": 0.45,
    "estimand_mismatch": 0.65,
}

RISK_COMPONENTS = list(DEFAULT_RISK_WEIGHTS)


def _with_design_priors(metrics: pd.DataFrame) -> pd.DataFrame:
    df = metrics.copy()
    catalog = design_catalog_frame()[["name", "operational_cost", "estimand_mismatch"]].rename(
        columns={"name": "design"}
    )
    df = df.merge(catalog, on="design", how="left", suffixes=("", "_prior"))
    for col in ["operational_cost", "estimand_mismatch"]:
        prior_col = f"{col}_prior"
        if col not in metrics.columns:
            df[col] = df[prior_col]
        else:
            df[col] = df[col].fillna(df[prior_col])
        df = df.drop(columns=[prior_col])
    return df


def _ensure_risk_components(metrics: pd.DataFrame) -> pd.DataFrame:
    """Create the paper-facing risk components, keeping older columns as aliases."""
    df = metrics.copy()
    if "exposure_distance" not in df.columns:
        if "abs_bias" in df.columns:
            df["exposure_distance"] = df["abs_bias"]
        elif "contamination_risk" in df.columns:
            df["exposure_distance"] = df["contamination_risk"]
        else:
            df["exposure_distance"] = 0.0
    if "assignment_unit_variance" not in df.columns:
        if "assignment_unit_sd" in df.columns:
            df["assignment_unit_variance"] = df["assignment_unit_sd"] ** 2
        elif "sd" in df.columns:
            df["assignment_unit_variance"] = df["sd"] ** 2
        else:
            df["assignment_unit_variance"] = 0.0
    if "planning_mde" not in df.columns:
        if "mde" in df.columns:
            df["planning_mde"] = df["mde"]
        elif "mean_mde" in df.columns:
            df["planning_mde"] = df["mean_mde"]
        else:
            df["planning_mde"] = 0.0
    if "contamination_risk" not in df.columns:
        scale = df.get("true_launch_effect", pd.Series([1.0] * len(df))).abs().clip(lower=1e-6)
        df["contamination_risk"] = df["exposure_distance"] / scale
    return df


def design_recommendations(
    metrics: pd.DataFrame,
    weights: dict[str, float] | None = None,
    robust_over_mechanisms: bool = True,
) -> pd.DataFrame:
    """Rank designs by the paper's interference-aware design-risk score."""
    weights = weights or DEFAULT_RISK_WEIGHTS
    df = _ensure_risk_components(_with_design_priors(metrics))
    for col in RISK_COMPONENTS:
        if col not in df.columns:
            raise KeyError(f"Missing design-risk component: {col}")
        scale = df[col].replace([np.inf, -np.inf], np.nan).abs().max()
        df[f"{col}_norm"] = 0.0 if not scale or np.isnan(scale) else df[col] / scale
    df["design_risk"] = 0.0
    for col, weight in weights.items():
        df["design_risk"] += weight * df[f"{col}_norm"]
    if robust_over_mechanisms and "theta_id" in df.columns:
        group_cols = [
            col for col in ["case", "open_bandit_slice", "design"] if col in df.columns
        ]
        if "design" not in group_cols:
            group_cols = ["design"]
        theta_counts = df.groupby(group_cols)["theta_id"].nunique().rename("n_mechanisms")
        worst_idx = df.groupby(group_cols)["design_risk"].idxmax()
        worst = df.loc[worst_idx].copy()
        worst = worst.merge(theta_counts, on=group_cols, how="left")
        worst["worst_case_theta_id"] = worst["theta_id"]
        if "mechanism_label" in worst.columns:
            worst["worst_case_mechanism_label"] = worst["mechanism_label"]
        for col in ["gamma_g", "gamma_b", "lambda", "locality"]:
            if col in worst.columns:
                worst[f"worst_case_{col}"] = worst[col]
        worst["robust_selection_rule"] = "max_theta_design_risk"
        return worst.sort_values(
            ["design_risk", "exposure_distance", "planning_mde"]
        ).reset_index(drop=True)
    return df.sort_values(
        ["design_risk", "exposure_distance", "planning_mde"]
    ).reset_index(drop=True)


def pareto_frontier(metrics: pd.DataFrame) -> pd.DataFrame:
    """Return designs not componentwise dominated on the risk components."""
    df = _ensure_risk_components(metrics)
    if "design_risk" not in df.columns:
        df = design_recommendations(df)
    dominated = []
    values = df[RISK_COMPONENTS].to_numpy(float)
    for i, row in enumerate(values):
        other = np.delete(values, i, axis=0)
        is_dominated = np.any(np.all(other <= row, axis=1) & np.any(other < row, axis=1))
        dominated.append(bool(is_dominated))
    out = df.assign(is_pareto_frontier=np.logical_not(dominated))
    return out[out["is_pareto_frontier"]].sort_values("design_risk").reset_index(drop=True)


def uncertainty_shortlist(
    ranked: pd.DataFrame,
    epsilon: float | None = None,
    epsilon_fraction: float = 0.10,
) -> pd.DataFrame:
    """Return designs within the implementation's planning-tolerance band."""
    df = ranked.sort_values("design_risk").copy()
    used_default = epsilon is None
    if epsilon is None:
        epsilon = epsilon_fraction * max(float(df["design_risk"].min()), 1e-12)
    threshold = float(df["design_risk"].min() + 2 * epsilon)
    df["shortlist_epsilon"] = epsilon
    df["shortlist_epsilon_fraction"] = epsilon_fraction
    df["shortlist_epsilon_method"] = (
        "pre_specified_fraction_of_best_robust_risk" if used_default else "user_supplied"
    )
    df["shortlist_threshold"] = threshold
    df["in_uncertainty_shortlist"] = df["design_risk"] <= threshold
    return df[df["in_uncertainty_shortlist"]].reset_index(drop=True)


def finite_selector_diagnostic(
    estimated_ranked: pd.DataFrame,
    oracle_ranked: pd.DataFrame,
) -> pd.DataFrame:
    """Check the finite-design theorem using an oracle risk proxy from known-truth simulation."""
    est = estimated_ranked[["design", "design_risk"]].rename(
        columns={"design_risk": "estimated_design_risk"}
    )
    oracle = oracle_ranked[["design", "design_risk"]].rename(
        columns={"design_risk": "oracle_design_risk"}
    )
    merged = est.merge(oracle, on="design", how="inner")
    epsilon = float((merged["estimated_design_risk"] - merged["oracle_design_risk"]).abs().max())
    selected = merged.sort_values("estimated_design_risk").iloc[0]
    oracle_best = merged.sort_values("oracle_design_risk").iloc[0]
    excess = float(selected["oracle_design_risk"] - oracle_best["oracle_design_risk"])
    return pd.DataFrame(
        [
            {
                "selected_design": selected["design"],
                "oracle_best_design": oracle_best["design"],
                "epsilon": epsilon,
                "two_epsilon_bound": 2 * epsilon,
                "oracle_excess_risk": excess,
                "bound_holds": excess <= 2 * epsilon + 1e-12,
            }
        ]
    )
