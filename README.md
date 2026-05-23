# Choosing Online Experiment Designs under Interference

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![Workflow: Notebook first](https://img.shields.io/badge/Workflow-notebook--first-purple.svg)
![Generated artifacts: ignored](https://img.shields.io/badge/Generated%20artifacts-ignored-lightgrey.svg)

![Robust design-selection workflow](img/info.png)

Notebook-first code for the paper **Choosing Online Experiment Designs under Interference in Ads, Recommendations, and Member-Experience Systems**.

The notebooks are the public face of the analysis. Reusable code lives in `src/`, while generated tables and figures are written to `artifacts/` and ignored by git. The analysis follows the paper's robust design selector: each domain notebook estimates exposure distance, assignment-unit variance, planning MDE, contamination or carryover risk, operational cost, and estimand mismatch before writing a design-risk ranking, Pareto frontier, and uncertainty shortlist. Open Bandit additionally injects a known-propensity support stress into the contamination and estimand-mismatch components using the inverse-propensity effective sample share.

The manuscript uses three representative empirical blocks in the main body, all anchored in real public datasets: Criteo, Open Bandit, and KuaiRand. The cross-domain notebook creates the main summary figure from those real-data cases. Synthetic selector diagnostics, the MovieLens graph simulation, dataset readiness checks, and controlled theorem stress tests support the appendix.

## Setup

```bash
uv sync
uv run jupyter lab
```

Then open the notebooks in order from `notebooks/`.

The public datasets are not committed to this repository. Put local dataset downloads outside the repo or in a local `data/` directory, which is ignored by git.

## Notebook Map

| Notebook | Dataset basis | Paper placement | Role |
|---|---|---|---|
| `00_dataset_readiness.ipynb` | Criteo, Open Bandit, KuaiRand, MovieLens, MIND, KuaiRec metadata | Appendix reproducibility | Verifies local data availability and inspects archives without unpacking large files. |
| `01_design_selector_synthetic_demo.ipynb` | Synthetic platform simulator | Appendix theorem diagnostic | Demonstrates the finite-design selector, theorem diagnostic, Pareto frontier, uncertainty shortlist, and interference-strength sensitivity when the true launch effect is known. |
| `02_criteo_ads_interference_designs.ipynb` | Criteo Uplift | Main body | Calibrates an ads budget-spillover simulation from randomized ads incrementality data and writes ads-specific selector artifacts. |
| `03_open_bandit_known_propensity_designs.ipynb` | Open Bandit `random/men` and `bts/men` | Main body plus appendix baseline | Uses known propensities to record inverse-propensity diagnostics. The adaptive `bts/men` slice anchors the main support-stress case, while `random/men` is preserved as a randomized baseline. |
| `04_kuairand_recommendation_carryover_designs.ipynb` | KuaiRand-Pure | Main body | Calibrates a sequential recommendation/member-experience carryover simulation and writes carryover-specific selector artifacts. |
| `05_graph_interference_design_frontier.ipynb` | MovieLens 32M | Appendix detail | Uses a user-item graph substrate to compare graph-interference designs under known ground truth and writes graph-specific selector artifacts. |
| `06_cross_domain_design_recommendations.ipynb` | Outputs from notebooks 01-05 | Main-body synthesis plus appendix tables | Produces paper-facing recommendations from the three real-data cases and preserves synthetic and graph cases for appendix tables. |
| `07_theory_stress_tests_appendix.ipynb` | Controlled mathematical stress tests | Appendix | Produces the theory-to-artifact map and stress-tests the transport bound, minimax sharpness, finite-catalog approximation, and assignment-unit MDE calculation. |

## Theorem Coverage

| Paper result | Primary empirical artifact | Placement |
|---|---|---|
| Lemma A.1, no uniformly best default design | `06_design_recommendations.csv` in the main text; `01_interference_sensitivity.png` in the appendix | Main body plus appendix diagnostic |
| Theorem 4.1, transport-based bias bound | `07_transport_bias_bound_stress.png`; `07_transport_bias_bound_stress.csv` | Appendix |
| Theorem 4.2, minimax optimality of exposure geometry | `07_minimax_transport_optimality.png`; `07_minimax_transport_optimality.csv` | Appendix, with brief main-body mention |
| Proposition 4.3, finite catalog approximation | `07_catalog_approximation_sensitivity.png`; `07_catalog_approximation_sensitivity.csv` | Appendix |
| Proposition 4.4, design-regime threshold | `01_regime_transition_design_risks.png`; `01_regime_reversal_design_map.png`; matching CSV files | Main-body regime diagnostic plus appendix |
| Theorem 4.5, robust selector and shortlist certification | `06_cross_domain_uncertainty_shortlists.csv` in the main text; `01_selector_theorem_diagnostic.png` and `01_selector_theorem_diagnostic.csv` in the appendix | Main body plus appendix diagnostic |
| Assignment-unit MDE planning equation | `07_assignment_unit_mde_grid.png`; `07_assignment_unit_mde_grid.csv` | Appendix |
| Semi-synthetic exposure-mechanism validation | `02_*`, `03_*`, `04_*`, and real-data subset of `06_*` in the main text; `05_*` in the appendix | Main body plus appendix details |

## Outputs

Running the notebooks writes:

- `artifacts/tables/*.csv`
- `artifacts/figures/*.png`

Important paper-facing tables include:

- `01_selector_theorem_diagnostic.csv`
- `02_criteo_ads_design_scores.csv`
- `03_open_bandit_propensity_diagnostics.csv`
- `03_open_bandit_design_scores.csv`
- `04_kuairand_design_scores.csv`
- `*_pareto_frontier.csv`
- `*_uncertainty_shortlist.csv`
- `06_main_body_real_data_design_scores.csv`
- `06_design_recommendations.csv`
- `06_appendix_all_case_design_recommendations.csv`
- `06_cross_domain_pareto_frontiers.csv`
- `06_cross_domain_uncertainty_shortlists.csv`
- `07_theory_to_experiment_map.csv`
- `07_transport_bias_bound_stress.csv`
- `07_minimax_transport_optimality.csv`
- `07_catalog_approximation_sensitivity.csv`
- `07_assignment_unit_mde_grid.csv`

These files are generated artifacts and are intentionally not committed.

## Repository Layout

```text
.
├── README.md
├── LICENSE
├── pyproject.toml
├── uv.lock
├── notebooks/        # notebook-first experiment workflow
├── src/              # reusable loaders, simulators, selectors, tables, and figures
├── img/              # README/paper-facing static images
└── artifacts/        # generated figures, tables, and workspace files ignored by git
```

## License

This code is released under the MIT License. See `LICENSE`.
