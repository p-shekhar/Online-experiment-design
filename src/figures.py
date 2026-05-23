from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update(
    {
        "figure.dpi": 130,
        "savefig.dpi": 300,
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
    }
)


def save_figure(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    print(f"[saved] {path}")
    return path


SHORT_DESIGN_LABELS = {
    "user_randomization": "User rand.",
    "cluster_randomization": "Cluster rand.",
    "switchback": "Switchback",
    "budget_split": "Budget split",
    "two_stage_saturation": "Two-stage",
    "mixed_randomization": "Mixed rand.",
}


DESIGN_CODES = {
    "user_randomization": "U",
    "cluster_randomization": "C",
    "switchback": "S",
    "budget_split": "B",
    "two_stage_saturation": "T",
    "mixed_randomization": "M",
}


CASE_LABELS = {
    "criteo_ads_budget_interference": "Criteo ads",
    "open_bandit_bts_known_propensity": "Open Bandit bts/men",
    "open_bandit_random_uniform_baseline": "Open Bandit random/men",
    "kuairand_member_experience_carryover": "KuaiRand carryover",
    "movielens_graph_interference": "MovieLens graph",
    "budget_spillover": "Synthetic budget",
}


SCATTER_LABEL_OFFSETS = {
    "user_randomization": (12, 24),
    "cluster_randomization": (10, 14),
    "switchback": (12, 2),
    "budget_split": (12, -24),
    "two_stage_saturation": (-58, 34),
    "mixed_randomization": (14, -18),
}


def plot_design_risk(summary: pd.DataFrame, title: str, path: Path) -> plt.Figure:
    df = summary.sort_values("design_risk")
    fig, ax = plt.subplots(figsize=(11, 5.8))
    colors = np.where(df["design"] == df.iloc[0]["design"], "#1f77b4", "#8a8a8a")
    ax.barh(df["design"], df["design_risk"], color=colors)
    ax.invert_yaxis()
    ax.set_xlabel("Design risk score, lower is better")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.25)
    save_figure(fig, path)
    return fig


def plot_bias_variance_frontier(summary: pd.DataFrame, title: str, path: Path) -> plt.Figure:
    x_col = "exposure_distance" if "exposure_distance" in summary.columns else "abs_bias"
    y_col = (
        "assignment_unit_variance" if "assignment_unit_variance" in summary.columns else "sd"
    )
    color_col = "planning_mde" if "planning_mde" in summary.columns else "mde"
    fig, ax = plt.subplots(figsize=(9.8, 6.4))
    scatter = ax.scatter(
        summary[x_col],
        summary[y_col],
        s=230 * (summary["operational_cost"] + 0.24),
        c=summary[color_col],
        cmap="viridis",
        edgecolor="black",
        linewidth=0.8,
    )
    color_values = summary[color_col].astype(float)
    color_midpoint = float((color_values.min() + color_values.max()) / 2)
    for _, row in summary.iterrows():
        design = row["design"]
        code = DESIGN_CODES.get(design, design[:1].upper())
        text_color = "black" if float(row[color_col]) > color_midpoint else "white"
        ax.text(
            row[x_col],
            row[y_col],
            code,
            fontsize=9,
            fontweight="bold",
            ha="center",
            va="center",
            color=text_color,
        )
    ax.set_xlabel("Exposure-distance proxy")
    ax.set_ylabel("Assignment-unit variance")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.margins(x=0.04, y=0.14)
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor="#efefef",
            markeredgecolor="black",
            label=f"{DESIGN_CODES[name]} = {label}",
        )
        for name, label in SHORT_DESIGN_LABELS.items()
        if name in set(summary["design"])
    ]
    ax.legend(
        handles=handles,
        title="Design codes",
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=3,
        frameon=True,
        framealpha=0.9,
        fontsize=8,
        title_fontsize=9,
    )
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Planning MDE")
    save_figure(fig, path)
    return fig


def plot_open_bandit_propensity_comparison(
    calibration: pd.DataFrame,
    diagnostics: pd.DataFrame,
    path: Path,
) -> plt.Figure:
    """Compare propensity stability across Open Bandit logging policies."""
    cal = calibration.copy()
    cal["slice"] = cal["behavior_policy"] + "/" + cal["campaign"]
    diag = diagnostics[diagnostics["clip_cap"].astype(str) == "none"].copy()
    diag["slice"] = diag["behavior_policy"] + "/" + diag["campaign"]
    df = cal.merge(
        diag[["slice", "weight_max", "weight_sd"]],
        on="slice",
        how="left",
    )
    x = np.arange(len(df))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))

    axes[0].bar(x, df["ips_ess_share"], color="#2a9d8f")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(df["slice"], rotation=15, ha="right")
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("IPS effective sample-size share")
    axes[0].set_title("Effective support")
    axes[0].grid(axis="y", alpha=0.25)

    width = 0.36
    axes[1].bar(x - width / 2, df["max_propensity"], width, label="max propensity")
    axes[1].bar(x + width / 2, df["min_propensity"], width, label="min propensity")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(df["slice"], rotation=15, ha="right")
    axes[1].set_ylabel("Logged propensity")
    axes[1].set_title("Propensity range")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend()

    fig.suptitle("Open Bandit logging-policy contrast", y=1.03)
    save_figure(fig, path)
    return fig


def plot_case_recommendations(cases: pd.DataFrame, path: Path) -> plt.Figure:
    winners = cases.sort_values("design_risk").groupby("case", as_index=False).first()
    winners["case_label"] = winners["case"].map(CASE_LABELS).fillna(winners["case"])
    winners["design_label"] = winners["design"].map(SHORT_DESIGN_LABELS).fillna(
        winners["design"]
    )
    fig, ax = plt.subplots(figsize=(10.8, 5.3))
    ax.barh(winners["case_label"], winners["design_risk"], color="#2a9d8f")
    for idx, row in winners.iterrows():
        ax.text(row["design_risk"] + 0.025, idx, row["design_label"], va="center")
    ax.invert_yaxis()
    ax.set_xlabel("Winning design risk score")
    ax.set_title("Recommended design by empirical scenario")
    ax.set_xlim(0, winners["design_risk"].max() * 1.22)
    ax.grid(axis="x", alpha=0.25)
    save_figure(fig, path)
    return fig


def plot_sensitivity_grid(results: pd.DataFrame, path: Path) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 6))
    for design, df in results.groupby("design"):
        ax.plot(df["interference_strength"], df["design_risk"], marker="o", label=design)
    ax.set_xlabel("Interference strength")
    ax.set_ylabel("Design risk score")
    ax.set_title("Design preference changes as interference increases")
    ax.grid(alpha=0.25)
    ax.legend(ncol=2)
    save_figure(fig, path)
    return fig


def plot_regime_reversal_map(results: pd.DataFrame, path: Path) -> plt.Figure:
    df = results.copy()
    df["design_label"] = df["design"].map(SHORT_DESIGN_LABELS).fillna(df["design"])
    regimes = (
        df[["regime_order", "regime"]]
        .drop_duplicates()
        .sort_values("regime_order")
    )
    designs = [
        "user_randomization",
        "mixed_randomization",
        "cluster_randomization",
        "switchback",
        "budget_split",
        "two_stage_saturation",
    ]
    fig, ax = plt.subplots(figsize=(11, 6.2))
    for design in designs:
        sub = df[df["design"] == design].sort_values("regime_order")
        ax.plot(
            sub["regime_order"],
            sub["design_risk"],
            marker="o",
            linewidth=2.0,
            label=SHORT_DESIGN_LABELS.get(design, design),
        )
    winners = df[df["is_selected"]].sort_values("regime_order")
    ax.scatter(
        winners["regime_order"],
        winners["design_risk"],
        s=160,
        facecolors="none",
        edgecolors="black",
        linewidths=2.0,
        label="selected design",
        zorder=5,
    )
    for _, row in winners.iterrows():
        ax.annotate(
            SHORT_DESIGN_LABELS.get(row["design"], row["design"]),
            xy=(row["regime_order"], row["design_risk"]),
            xytext=(0, 14),
            textcoords="offset points",
            ha="center",
            fontsize=10,
            weight="bold",
        )
    ax.set_xticks(regimes["regime_order"])
    ax.set_xticklabels(regimes["regime"], rotation=12, ha="right")
    ax.set_ylabel("Design risk score")
    ax.set_xlabel("Exposure-mechanism regime")
    ax.set_title("Regime reversal under uncertain exposure mechanisms")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=2, fontsize=9)
    save_figure(fig, path)
    return fig


def plot_regime_transition_curves(results: pd.DataFrame, path: Path) -> plt.Figure:
    df = results.copy()
    designs = [
        "user_randomization",
        "mixed_randomization",
        "cluster_randomization",
        "switchback",
        "budget_split",
        "two_stage_saturation",
    ]
    colors = {
        "user_randomization": "#4c78a8",
        "mixed_randomization": "#f58518",
        "cluster_randomization": "#54a24b",
        "switchback": "#b279a2",
        "budget_split": "#e45756",
        "two_stage_saturation": "#72b7b2",
    }

    fig, ax = plt.subplots(figsize=(11.2, 6.4))
    for design in designs:
        sub = df[df["design"] == design].sort_values("gamma")
        ax.plot(
            sub["gamma"],
            sub["design_risk"],
            linewidth=2.4,
            color=colors.get(design),
            label=SHORT_DESIGN_LABELS.get(design, design),
        )

    selected = df[df["is_selected"]].sort_values("gamma")
    ax.plot(
        selected["gamma"],
        selected["design_risk"],
        color="black",
        linestyle=(0, (5, 3)),
        linewidth=2.4,
        alpha=0.88,
        label="selected envelope",
        zorder=6,
    )

    transitions = selected[selected["design"].ne(selected["design"].shift())]
    y_max = float(df["design_risk"].max())
    y_min = float(df["design_risk"].min())
    for _, row in transitions.iloc[1:].iterrows():
        ax.axvline(row["gamma"], color="black", linestyle="--", linewidth=1.0, alpha=0.35)
    for _, row in transitions.iterrows():
        label = SHORT_DESIGN_LABELS.get(row["design"], row["design"])
        label_offsets = {
            "user_randomization": (12, 26),
            "mixed_randomization": (16, 32),
            "cluster_randomization": (14, 34),
            "switchback": (12, 30),
        }
        ax.annotate(
            label,
            xy=(row["gamma"], row["design_risk"]),
            xytext=label_offsets.get(row["design"], (10, 24)),
            textcoords="offset points",
            fontsize=9,
            weight="bold",
            color="black",
            bbox={
                "boxstyle": "round,pad=0.20",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.86,
            },
        )

    ax.axvspan(0.0, 0.35, color="#4c78a8", alpha=0.06)
    ax.axvspan(0.35, 0.65, color="#f58518", alpha=0.06)
    ax.axvspan(0.65, 0.90, color="#54a24b", alpha=0.06)
    ax.axvspan(0.90, 1.0, color="#b279a2", alpha=0.06)
    phase_box = {
        "boxstyle": "round,pad=0.18",
        "facecolor": "white",
        "edgecolor": "none",
        "alpha": 0.75,
    }
    ax.text(0.16, y_max * 0.97, "row-local", ha="center", fontsize=9, bbox=phase_box)
    ax.text(0.50, y_max * 0.97, "mixed spillover", ha="center", fontsize=9, bbox=phase_box)
    ax.text(0.77, y_max * 0.97, "clustered spillover", ha="center", fontsize=9, bbox=phase_box)
    ax.text(0.95, y_max * 0.97, "carryover", ha="center", fontsize=9, bbox=phase_box)

    ax.set_xlabel(r"Exposure-mechanism intensity $\gamma$")
    ax.set_ylabel("Robust planning risk, lower is better")
    ax.set_title("Regime transitions in robust experiment-design selection")
    ax.set_xlim(0, 1)
    ax.set_ylim(y_min * 0.90, y_max * 1.05)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(
        ncol=3,
        frameon=True,
        framealpha=0.95,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
    )
    save_figure(fig, path)
    return fig


def plot_selector_guarantee(
    plan_ranked: pd.DataFrame,
    oracle_ranked: pd.DataFrame,
    path: Path,
) -> plt.Figure:
    df = (
        plan_ranked[["design", "design_risk"]]
        .rename(columns={"design_risk": "planning risk"})
        .merge(
            oracle_ranked[["design", "design_risk"]].rename(
                columns={"design_risk": "oracle risk"}
            ),
            on="design",
        )
        .sort_values("oracle risk")
    )
    x = np.arange(len(df))
    width = 0.38
    fig, ax = plt.subplots(figsize=(11, 5.8))
    ax.bar(x - width / 2, df["planning risk"], width, label="planning risk")
    ax.bar(x + width / 2, df["oracle risk"], width, label="oracle risk")
    ax.set_xticks(x)
    ax.set_xticklabels(df["design"], rotation=25, ha="right")
    ax.set_ylabel("Design risk score")
    ax.set_title("Finite-design selector diagnostic")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    save_figure(fig, path)
    return fig


def plot_transport_bound_checks(table: pd.DataFrame, path: Path) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7.5, 6))
    ax.scatter(table["transport_bound"], table["observed_bias"], s=90, color="#457b9d")
    high = max(table["transport_bound"].max(), table["observed_bias"].max()) * 1.05
    ax.plot([0, high], [0, high], linestyle="--", color="black", label="bound equality")
    label_offsets = {
        "small_shift": (8, 14),
        "medium_shift": (8, -18),
        "large_shift": (10, -18),
        "steeper_response": (10, 8),
        "flat_response": (10, -6),
    }
    for _, row in table.iterrows():
        offset = label_offsets.get(row["scenario"], (5, 4))
        ax.annotate(
            row["scenario"],
            (row["transport_bound"], row["observed_bias"]),
            xytext=offset,
            textcoords="offset points",
            fontsize=10,
        )
    ax.set_xlabel("Transport bound, L times W1")
    ax.set_ylabel("Observed exposure-response bias")
    ax.set_title("Transport-based bias bound stress test")
    ax.grid(alpha=0.25)
    ax.legend()
    save_figure(fig, path)
    return fig


def plot_minimax_transport_check(table: pd.DataFrame, path: Path) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for lip, df in table.groupby("lipschitz_constant"):
        ax.plot(df["delta"], df["attainment_ratio"], marker="o", label=f"L={lip:g}")
    ax.axhline(1.0, linestyle="--", color="black", label="minimax equality")
    ax.set_xlabel("Exposure shift")
    ax.set_ylabel("Attained gap / minimax penalty")
    ax.set_title("Minimax optimality of exposure geometry")
    ax.set_ylim(0.0, 1.08)
    ax.grid(alpha=0.25)
    ax.legend()
    save_figure(fig, path)
    return fig


def plot_catalog_approximation(table: pd.DataFrame, path: Path) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.plot(table["catalog_size"], table["approximation_gap"], marker="o", label="catalog gap")
    ax.plot(table["catalog_size"], table["lipschitz_eta_bound"], marker="s", label="L eta bound")
    ax.set_xlabel("Catalog size")
    ax.set_ylabel("Risk gap")
    ax.set_title("Finite catalog approximation stress test")
    ax.grid(alpha=0.25)
    ax.legend()
    save_figure(fig, path)
    return fig


def plot_assignment_unit_mde(table: pd.DataFrame, path: Path) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9, 5.8))
    for design, df in table.groupby("design"):
        ax.plot(
            df["weeks"],
            df["planning_mde"],
            marker="o",
            label=SHORT_DESIGN_LABELS.get(design, design),
        )
    ax.set_xlabel("Experiment duration, weeks")
    ax.set_ylabel("Planning MDE")
    ax.set_title("MDE depends on effective assignment units")
    ax.grid(alpha=0.25)
    ax.legend(ncol=2)
    save_figure(fig, path)
    return fig
