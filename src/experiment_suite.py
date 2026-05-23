from __future__ import annotations

import pandas as pd
import numpy as np

from designs import DEFAULT_RISK_WEIGHTS, design_recommendations, finite_selector_diagnostic
from simulators import SimulationConfig, run_design_simulation, summarize_simulation


def run_recommended_synthetic_scenarios(n_reps: int = 120) -> pd.DataFrame:
    scenarios = [
        ("low_interference", 0.08, 0.01, 0.005),
        ("budget_spillover", 0.08, 0.07, 0.015),
        ("temporal_carryover", 0.08, 0.03, 0.065),
        ("network_spillover", 0.08, 0.10, 0.01),
    ]
    rows = []
    for idx, (case, direct, spillover, carryover) in enumerate(scenarios):
        config = SimulationConfig(
            n_users=2200,
            n_clusters=55,
            n_periods=16,
            direct_effect=direct,
            spillover_strength=spillover,
            carryover_strength=carryover,
            noise_sd=0.25,
            seed=7000 + idx,
        )
        summary = summarize_simulation(run_design_simulation(config, n_reps=n_reps))
        ranked = design_recommendations(summary)
        ranked["case"] = case
        rows.append(ranked)
    return pd.concat(rows, ignore_index=True)


def run_interference_sensitivity(n_reps: int = 80) -> pd.DataFrame:
    rows = []
    for level in [0.0, 0.025, 0.05, 0.075, 0.10, 0.15]:
        config = SimulationConfig(
            n_users=1800,
            n_clusters=45,
            n_periods=12,
            direct_effect=0.08,
            spillover_strength=level,
            carryover_strength=0.02,
            noise_sd=0.25,
            seed=9000 + int(level * 1000),
        )
        summary = summarize_simulation(run_design_simulation(config, n_reps=n_reps))
        ranked = design_recommendations(summary)
        ranked["interference_strength"] = level
        rows.append(ranked)
    return pd.concat(rows, ignore_index=True)


def run_regime_reversal_diagnostic() -> pd.DataFrame:
    """Construct a clean mechanism sweep for the design-regime theorem.

    The public-data experiments instantiate several realistic regions of the
    design space. This diagnostic isolates the theorem-level regime map by
    holding the catalog and risk components fixed while varying the exposure
    mechanism from row-local effects through mixed spillover, clustered
    spillover, and carryover-dominant interference.
    """
    regimes = [
        ("weak row-local", 0.00, 0.00),
        ("medium mixed spillover", 0.35, 0.05),
        ("strong cluster spillover", 0.85, 0.05),
        ("high carryover", 0.10, 0.90),
    ]
    rows: list[dict[str, float | str]] = []
    for order, (regime, spillover, carryover) in enumerate(regimes, start=1):
        for design in [
            "user_randomization",
            "mixed_randomization",
            "cluster_randomization",
            "switchback",
            "budget_split",
            "two_stage_saturation",
        ]:
            if design == "user_randomization":
                exposure_distance = 0.08 + 1.40 * spillover + 0.75 * carryover
                assignment_unit_variance = 0.025
                planning_mde = 0.10
                contamination_risk = 0.04 + 1.50 * spillover + 0.95 * carryover
                operational_cost = 0.10
                estimand_mismatch = 0.05 + 0.90 * spillover + 0.65 * carryover
            elif design == "mixed_randomization":
                exposure_distance = (
                    0.15 + 0.50 * spillover + 0.55 * carryover + 0.45 * spillover**2
                )
                assignment_unit_variance = 0.060
                planning_mde = 0.18
                contamination_risk = (
                    0.08 + 0.50 * spillover + 0.75 * carryover + 0.55 * spillover**2
                )
                operational_cost = 0.50
                estimand_mismatch = (
                    0.10 + 0.25 * spillover + 0.40 * carryover + 0.30 * spillover**2
                )
            elif design == "cluster_randomization":
                exposure_distance = 0.22 + 0.05 * spillover + 0.55 * carryover
                assignment_unit_variance = 0.120
                planning_mde = 0.30
                contamination_risk = 0.10 + 0.06 * spillover + 0.85 * carryover
                operational_cost = 0.35
                estimand_mismatch = 0.18 + 0.04 * spillover + 0.45 * carryover
            elif design == "switchback":
                exposure_distance = 0.35 + 0.80 * spillover
                assignment_unit_variance = 0.120
                planning_mde = 0.28
                contamination_risk = 0.08 + 0.70 * spillover + 0.01 * carryover
                operational_cost = 0.45
                estimand_mismatch = 0.10 + 0.50 * spillover
            elif design == "budget_split":
                exposure_distance = 0.20 + 0.15 * spillover + 0.60 * carryover
                assignment_unit_variance = 0.160
                planning_mde = 0.35
                contamination_risk = 0.12 + 0.12 * spillover + 0.75 * carryover
                operational_cost = 0.70
                estimand_mismatch = 0.15 + 0.08 * spillover + 0.42 * carryover
            elif design == "two_stage_saturation":
                exposure_distance = 0.18 + 0.30 * spillover + 0.55 * carryover
                assignment_unit_variance = 0.090
                planning_mde = 0.20
                contamination_risk = 0.09 + 0.32 * spillover + 0.70 * carryover
                operational_cost = 0.80
                estimand_mismatch = 0.12 + 0.18 * spillover + 0.35 * carryover
            else:
                raise ValueError(f"Unknown design: {design}")

            rows.append(
                {
                    "regime_order": order,
                    "regime": regime,
                    "spillover_strength": spillover,
                    "carryover_strength": carryover,
                    "design": design,
                    "exposure_distance": exposure_distance,
                    "assignment_unit_variance": assignment_unit_variance,
                    "planning_mde": planning_mde,
                    "contamination_risk": contamination_risk,
                    "operational_cost": operational_cost,
                    "estimand_mismatch": estimand_mismatch,
                }
            )

    ranked_rows = []
    for _, group in pd.DataFrame(rows).groupby(["regime_order", "regime"], sort=True):
        ranked = design_recommendations(group)
        ranked_rows.append(ranked)
    out = pd.concat(ranked_rows, ignore_index=True)
    out["is_selected"] = out.groupby("regime_order")["design_risk"].rank(method="first") == 1
    return out


def run_continuous_regime_transition(n_points: int = 101) -> pd.DataFrame:
    """Continuous version of the design-regime threshold diagnostic.

    The sweep keeps the design catalog fixed and varies a one-dimensional
    mechanism-intensity parameter. Low values represent row-local effects,
    middle values represent increasingly shared spillover, and high values add
    carryover pressure. Component scales are fixed over the entire sweep so risk
    curves are comparable across gamma values.
    """
    designs = [
        "user_randomization",
        "mixed_randomization",
        "cluster_randomization",
        "switchback",
        "budget_split",
        "two_stage_saturation",
    ]
    rows: list[dict[str, float | str | bool]] = []
    for gamma in [round(float(x), 4) for x in np.linspace(0.0, 1.0, n_points)]:
        spillover = min(gamma / 0.72, 1.0)
        carryover = max(0.0, min((gamma - 0.70) / 0.30, 1.0))
        for design in designs:
            if design == "user_randomization":
                exposure_distance = 0.07 + 2.60 * spillover + 0.90 * carryover
                assignment_unit_variance = 0.025
                planning_mde = 0.10
                contamination_risk = 0.04 + 2.80 * spillover + 1.05 * carryover
                operational_cost = 0.10
                estimand_mismatch = 0.05 + 2.10 * spillover + 0.70 * carryover
            elif design == "mixed_randomization":
                exposure_distance = 0.10 + 0.08 * spillover + 0.50 * carryover + 1.50 * spillover**2
                assignment_unit_variance = 0.045
                planning_mde = 0.14
                contamination_risk = 0.055 + 0.10 * spillover + 0.75 * carryover + 1.50 * spillover**2
                operational_cost = 0.30
                estimand_mismatch = 0.065 + 0.05 * spillover + 0.40 * carryover + 1.125 * spillover**2
            elif design == "cluster_randomization":
                exposure_distance = 0.46 - 0.34 * spillover + 0.62 * carryover
                assignment_unit_variance = 0.12
                planning_mde = 0.28
                contamination_risk = 0.18 - 0.12 * spillover + 0.90 * carryover
                operational_cost = 0.35
                estimand_mismatch = 0.26 - 0.13 * spillover + 0.50 * carryover
            elif design == "switchback":
                exposure_distance = (
                    0.40 + 0.85 * spillover * (1.0 - 0.70 * carryover) + 0.02 * (1.0 - carryover)
                )
                assignment_unit_variance = 0.12
                planning_mde = 0.28
                contamination_risk = (
                    0.14 + 0.78 * spillover * (1.0 - 0.65 * carryover) + 0.02 * (1.0 - carryover)
                )
                operational_cost = 0.45
                estimand_mismatch = (
                    0.13 + 0.60 * spillover * (1.0 - 0.70 * carryover) + 0.03 * (1.0 - carryover)
                )
            elif design == "budget_split":
                exposure_distance = 0.20 + 0.15 * spillover + 0.62 * carryover
                assignment_unit_variance = 0.16
                planning_mde = 0.35
                contamination_risk = 0.12 + 0.12 * spillover + 0.75 * carryover
                operational_cost = 0.70
                estimand_mismatch = 0.15 + 0.08 * spillover + 0.42 * carryover
            elif design == "two_stage_saturation":
                exposure_distance = 0.24 + 0.28 * spillover + 0.58 * carryover
                assignment_unit_variance = 0.09
                planning_mde = 0.20
                contamination_risk = 0.09 + 0.30 * spillover + 0.72 * carryover
                operational_cost = 0.80
                estimand_mismatch = 0.12 + 0.18 * spillover + 0.36 * carryover
            else:
                raise ValueError(f"Unknown design: {design}")

            rows.append(
                {
                    "gamma": gamma,
                    "spillover_strength": spillover,
                    "carryover_strength": carryover,
                    "design": design,
                    "exposure_distance": exposure_distance,
                    "assignment_unit_variance": assignment_unit_variance,
                    "planning_mde": planning_mde,
                    "contamination_risk": contamination_risk,
                    "operational_cost": operational_cost,
                    "estimand_mismatch": estimand_mismatch,
                }
            )

    out = pd.DataFrame(rows)
    for component, weight in DEFAULT_RISK_WEIGHTS.items():
        scale = out[component].abs().max()
        out[f"{component}_norm"] = 0.0 if not scale else out[component] / scale
    out["design_risk"] = 0.0
    for component, weight in DEFAULT_RISK_WEIGHTS.items():
        out["design_risk"] += weight * out[f"{component}_norm"]

    out["rank"] = out.groupby("gamma")["design_risk"].rank(method="first")
    out["is_selected"] = out["rank"] == 1
    best = out.groupby("gamma")["design_risk"].transform("min")
    out["shortlist_threshold"] = best * 1.20
    out["in_uncertainty_shortlist"] = out["design_risk"] <= out["shortlist_threshold"]
    return out.sort_values(["gamma", "design_risk"]).reset_index(drop=True)


def run_selector_guarantee_demo(
    n_oracle_reps: int = 260,
    n_plan_reps: int = 45,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Known-truth selector check for the robust-selector theorem.

    The oracle risk uses many simulation replicates. The planning risk uses only
    the first n_plan_reps replicates, mimicking a noisy pre-experiment design
    study. The theorem diagnostic checks whether the selected design's oracle
    excess risk is below the empirical 2 epsilon uniform-risk bound.
    """
    config = SimulationConfig(
        n_users=2600,
        n_clusters=65,
        n_periods=18,
        direct_effect=0.08,
        spillover_strength=0.085,
        carryover_strength=0.025,
        noise_sd=0.25,
        seed=5100,
    )
    raw = run_design_simulation(config, n_reps=n_oracle_reps)
    oracle_ranked = design_recommendations(summarize_simulation(raw))
    plan_ranked = design_recommendations(summarize_simulation(raw[raw["rep"] < n_plan_reps]))
    diagnostic = finite_selector_diagnostic(plan_ranked, oracle_ranked)
    return plan_ranked, oracle_ranked, diagnostic
