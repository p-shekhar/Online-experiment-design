from __future__ import annotations

import numpy as np
import pandas as pd

from designs import mde_from_assignment_units


def theory_experiment_map() -> pd.DataFrame:
    """Map each formal result to the empirical artifact that stress-tests it."""
    rows = [
        {
            "paper_result": "Lemma A.1, no uniformly best default design",
            "empirical_stress_test": "Report real-data recommendations and changing uncertainty shortlists, with a synthetic mechanism-regime diagnostic showing winner reversal across exposure mechanisms.",
            "placement": "main body plus appendix diagnostic",
            "notebook": "06_cross_domain_design_recommendations.ipynb; 01_design_selector_synthetic_demo.ipynb",
            "tables": "06_design_recommendations.csv; 01_regime_transition_design_risks.csv",
            "figures": "06_cross_domain_design_recommendations.png; 01_regime_transition_design_risks.png",
        },
        {
            "paper_result": "Theorem 4.1, transport-based bias bound",
            "empirical_stress_test": "Check that observed exposure-response bias is below L times empirical W1.",
            "placement": "appendix",
            "notebook": "07_theory_stress_tests_appendix.ipynb",
            "tables": "07_transport_bias_bound_stress.csv",
            "figures": "07_transport_bias_bound_stress.png",
        },
        {
            "paper_result": "Theorem 4.2, minimax optimality of exposure geometry",
            "empirical_stress_test": "Construct Lipschitz responses that attain the transport penalty.",
            "placement": "appendix, with brief main-body mention",
            "notebook": "07_theory_stress_tests_appendix.ipynb",
            "tables": "07_minimax_transport_optimality.csv",
            "figures": "07_minimax_transport_optimality.png",
        },
        {
            "paper_result": "Proposition 4.3, finite catalog approximation",
            "empirical_stress_test": "Refine a catalog grid and verify the approximation gap is below L eta.",
            "placement": "appendix",
            "notebook": "07_theory_stress_tests_appendix.ipynb",
            "tables": "07_catalog_approximation_sensitivity.csv",
            "figures": "07_catalog_approximation_sensitivity.png",
        },
        {
            "paper_result": "Proposition 4.4, design-regime threshold",
            "empirical_stress_test": "Sweep exposure-mechanism regimes and show user, mixed, cluster, and switchback designs winning in different regimes.",
            "placement": "main-body regime-transition diagnostic",
            "notebook": "01_design_selector_synthetic_demo.ipynb",
            "tables": "01_regime_transition_design_risks.csv; 01_regime_reversal_design_map.csv",
            "figures": "01_regime_transition_design_risks.png; 01_regime_reversal_design_map.png",
        },
        {
            "paper_result": "Theorem 4.5, robust selector and certified shortlist",
            "empirical_stress_test": "Report real-data shortlists in the main body and compare noisy planning-risk selection with an oracle simulation in the appendix.",
            "placement": "main body plus appendix diagnostic",
            "notebook": "06_cross_domain_design_recommendations.ipynb; 01_design_selector_synthetic_demo.ipynb",
            "tables": "06_cross_domain_uncertainty_shortlists.csv; 01_selector_theorem_diagnostic.csv; 01_budget_spillover_uncertainty_shortlist.csv",
            "figures": "06_cross_domain_design_recommendations.png; 01_selector_theorem_diagnostic.png",
        },
        {
            "paper_result": "Planning MDE under assignment-unit variance",
            "empirical_stress_test": "Vary effective assignment units and assignment-unit variance.",
            "placement": "appendix",
            "notebook": "07_theory_stress_tests_appendix.ipynb",
            "tables": "07_assignment_unit_mde_grid.csv",
            "figures": "07_assignment_unit_mde_grid.png",
        },
        {
            "paper_result": "Semi-synthetic bias proxies under known launch effects",
            "empirical_stress_test": "Calibrate ads, recommendation, and carryover mechanisms from real public data in the main body; preserve graph simulations in the appendix.",
            "placement": "main body plus appendix details",
            "notebook": "02_criteo_ads_interference_designs.ipynb; 03_open_bandit_known_propensity_designs.ipynb; 04_kuairand_recommendation_carryover_designs.ipynb; 05_graph_interference_design_frontier.ipynb",
            "tables": "02_criteo_ads_design_scores.csv; 03_open_bandit_design_scores.csv; 04_kuairand_design_scores.csv; 05_movielens_graph_design_scores.csv",
            "figures": "02_criteo_ads_design_risk.png; 03_open_bandit_design_risk.png; 04_kuairand_design_risk.png; 05_movielens_graph_design_risk.png",
        },
    ]
    return pd.DataFrame(rows)


def empirical_w1(x: np.ndarray, y: np.ndarray) -> float:
    """One-dimensional empirical W1 using matched quantiles."""
    x = np.sort(np.asarray(x, dtype=float))
    y = np.sort(np.asarray(y, dtype=float))
    if len(x) == len(y):
        return float(np.mean(np.abs(x - y)))
    grid = np.linspace(0.0, 1.0, max(len(x), len(y)))
    return float(np.mean(np.abs(np.quantile(x, grid) - np.quantile(y, grid))))


def run_transport_bias_bound_stress(
    n: int = 5000,
    seed: int = 17,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    scenarios = [
        ("small_shift", 0.08, 1.0),
        ("medium_shift", 0.18, 1.0),
        ("large_shift", 0.32, 1.0),
        ("steeper_response", 0.18, 2.5),
        ("flat_response", 0.18, 0.45),
    ]
    rows = []
    base = rng.beta(2.0, 5.0, size=n)
    for name, shift, lip in scenarios:
        experimental = base
        launch = np.clip(base + shift, 0.0, 1.0)
        response_experimental = lip * experimental
        response_launch = lip * launch
        w1 = empirical_w1(experimental, launch)
        observed_bias = abs(response_experimental.mean() - response_launch.mean())
        bound = lip * w1
        rows.append(
            {
                "scenario": name,
                "lipschitz_constant": lip,
                "w1_exposure_distance": w1,
                "observed_bias": observed_bias,
                "transport_bound": bound,
                "bias_to_bound_ratio": observed_bias / bound if bound > 0 else np.nan,
                "bound_holds": observed_bias <= bound + 1e-12,
            }
        )
    return pd.DataFrame(rows)


def run_minimax_transport_optimality_check() -> pd.DataFrame:
    rows = []
    for delta in [0.025, 0.05, 0.10, 0.20, 0.35]:
        for lip in [0.5, 1.0, 2.0]:
            experimental = np.zeros(2000)
            launch = np.full(2000, delta)
            w1 = empirical_w1(experimental, launch)
            # r(x)=Lx is L-Lipschitz and attains the W1 dual for these ordered measures.
            attained_gap = abs((lip * experimental).mean() - (lip * launch).mean())
            minimax_penalty = lip * w1
            rows.append(
                {
                    "delta": delta,
                    "lipschitz_constant": lip,
                    "w1_exposure_distance": w1,
                    "attained_gap": attained_gap,
                    "minimax_penalty": minimax_penalty,
                    "attainment_ratio": attained_gap / minimax_penalty,
                }
            )
    return pd.DataFrame(rows)


def _risk_surface(x: np.ndarray) -> np.ndarray:
    return (x - 0.37) ** 2 + 0.12 * np.maximum(0.0, 0.58 - x) + 0.04 * np.sin(5 * x) ** 2


def run_catalog_approximation_sensitivity() -> pd.DataFrame:
    dense = np.linspace(0.0, 1.0, 20001)
    dense_risk = _risk_surface(dense)
    continuous_best = float(dense_risk.min())
    gradients = np.abs(np.gradient(dense_risk, dense))
    lipschitz_upper = float(gradients.max())
    rows = []
    for catalog_size in [3, 5, 7, 11, 21, 41]:
        grid = np.linspace(0.0, 1.0, catalog_size)
        risk = _risk_surface(grid)
        eta = 1.0 / (2.0 * (catalog_size - 1))
        gap = float(risk.min() - continuous_best)
        bound = lipschitz_upper * eta
        rows.append(
            {
                "catalog_size": catalog_size,
                "eta_net_radius": eta,
                "continuous_best_risk": continuous_best,
                "catalog_best_risk": float(risk.min()),
                "approximation_gap": gap,
                "lipschitz_eta_bound": bound,
                "bound_holds": gap <= bound + 1e-12,
            }
        )
    return pd.DataFrame(rows)


def run_assignment_unit_mde_grid() -> pd.DataFrame:
    rows = []
    design_units = [
        ("user_randomization", 2500, 0.18),
        ("cluster_randomization", 80, 0.28),
        ("switchback", 28, 0.24),
        ("budget_split", 24, 0.30),
        ("two_stage_saturation", 120, 0.26),
        ("mixed_randomization", 180, 0.25),
    ]
    for design, units_per_week, unit_sd in design_units:
        for weeks in [1, 2, 4, 8]:
            n_units = units_per_week * weeks
            rows.append(
                {
                    "design": design,
                    "weeks": weeks,
                    "effective_assignment_units": n_units,
                    "assignment_unit_sd": unit_sd,
                    "assignment_unit_variance": unit_sd**2,
                    "planning_mde": mde_from_assignment_units(unit_sd, n_units),
                }
            )
    return pd.DataFrame(rows)
