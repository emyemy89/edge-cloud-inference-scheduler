from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


PLOT_DIR = Path(__file__).resolve().parent / "figures"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

CONDITIONS = ["stable", "moderate", "high"]
LOADS = ["low_load", "medium_load", "high_load", "very_high_load"]
LOAD_LABELS = ["Low", "Medium", "High", "Very high"]
POLICIES = ["Always Edge", "Always Cloud", "Greedy", "XGBoost"]

STYLE = {
    "Always Edge": {"color": "#0072B2", "marker": "o"},
    "Always Cloud": {"color": "#D55E00", "marker": "s"},
    "Greedy": {"color": "#009E73", "marker": "^"},
    "XGBoost": {"color": "#CC79A7", "marker": "D"},
}

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.7,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "savefig.dpi": 300,
})


def save_figure(fig, name):
    fig.savefig(
        PLOT_DIR / f"{name}.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)


def scenario_label(condition, load):
    return f"{condition.title()}\n{LOAD_LABELS[LOADS.index(load)]}"


def plot_all_results(summary_rows, run_rows=None):
    df = pd.DataFrame(summary_rows)

    if df.empty:
        print("No results available for plotting.")
        return

    run_df = pd.DataFrame(run_rows) if run_rows else pd.DataFrame()

    plot_latency_heatmap(df)
    plot_xgboost_improvement(df)
    plot_latency_cost_tradeoff(df)
    plot_latency_degradation(df)
    plot_deadline_latency_tradeoff(df)
    plot_latency_variability(df, run_df)
    plot_utilization(df)
    plot_paired_latency_difference(run_df)
    plot_node_selection_distribution(df, run_df)

    print(f"Figures saved to {PLOT_DIR}")


def plot_latency_heatmap(df):
    """Compare mean latency across all policies and scenarios."""
    fig, axes = plt.subplots(
        1, 3, figsize=(12, 4.5), sharey=True
    )

    values = df["latency_mean"].to_numpy(dtype=float)
    positive = values[np.isfinite(values) & (values > 0)]

    if len(positive) == 0:
        plt.close(fig)
        return

    vmin = positive.min()
    vmax = positive.max()

    if vmin == vmax:
        vmax = vmin * 1.01

    for ax, condition in zip(axes, CONDITIONS):
        subset = df[df["network_condition"] == condition]
        matrix = (
            subset.pivot(
                index="policy",
                columns="load_level",
                values="latency_mean",
            )
            .reindex(index=POLICIES, columns=LOADS)
        )

        values = matrix.to_numpy(dtype=float)
        masked = np.ma.masked_invalid(values)

        image = ax.imshow(
            masked,
            aspect="auto",
            cmap="viridis",
            norm=LogNorm(vmin=vmin, vmax=vmax),
        )

        ax.set_title(condition.title())
        ax.set_xticks(range(len(LOADS)))
        ax.set_xticklabels(LOAD_LABELS)
        ax.set_yticks(range(len(POLICIES)))
        ax.set_yticklabels(POLICIES)
        ax.set_xlabel("Workload intensity")

        for i in range(len(POLICIES)):
            for j in range(len(LOADS)):
                value = values[i, j]
                if np.isfinite(value):
                    color = "white" if value > np.sqrt(vmin * vmax) else "black"
                    ax.text(
                        j, i, f"{value:.1f}",
                        ha="center", va="center",
                        fontsize=7, color=color,
                    )

    axes[0].set_ylabel("Scheduling policy")
    fig.colorbar(
        image, ax=axes, label="Mean latency (ms, logarithmic color scale)",
        shrink=0.85, pad=0.03,
    )
    fig.suptitle("Latency across workload and network conditions", y=1.03)
    fig.tight_layout()
    save_figure(fig, "01_latency_heatmap")


def plot_xgboost_improvement(df):
    """Positive values indicate lower latency with XGBoost than Greedy."""
    fig, axes = plt.subplots(
        1, 3, figsize=(10.5, 3.5), sharey=True
    )

    for ax, condition in zip(axes, CONDITIONS):
        subset = df[df["network_condition"] == condition]

        greedy = (
            subset[subset["policy"] == "Greedy"]
            .set_index("load_level")["latency_mean"]
            .reindex(LOADS)
        )
        xgb = (
            subset[subset["policy"] == "XGBoost"]
            .set_index("load_level")["latency_mean"]
            .reindex(LOADS)
        )

        improvement = 100 * (greedy - xgb) / greedy
        colors = [
            STYLE["XGBoost"]["color"] if value >= 0
            else STYLE["Always Cloud"]["color"]
            for value in improvement
        ]

        ax.bar(
            range(len(LOADS)),
            improvement,
            color=colors,
            width=0.65,
            edgecolor="white",
            linewidth=0.5,
        )
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(range(len(LOADS)))
        ax.set_xticklabels(LOAD_LABELS)
        ax.set_title(condition.title())
        ax.set_xlabel("Workload intensity")
        ax.grid(axis="y", linestyle=":", linewidth=0.6)
        ax.set_axisbelow(True)

    axes[0].set_ylabel("Latency reduction vs Greedy (%)")
    fig.suptitle(
        "Incremental latency benefit of XGBoost over Greedy", y=1.04
    )
    fig.tight_layout()
    save_figure(fig, "02_xgboost_improvement")


def plot_latency_cost_tradeoff(df):
    """Lower-left points indicate lower latency and lower cost."""
    fig, axes = plt.subplots(
        1, 3, figsize=(11, 3.8), sharey=True
    )

    for ax, condition in zip(axes, CONDITIONS):
        subset = df[df["network_condition"] == condition]

        for policy in POLICIES:
            part = subset[subset["policy"] == policy]

            ax.scatter(
                part["cost_mean"],
                part["latency_mean"],
                color=STYLE[policy]["color"],
                marker=STYLE[policy]["marker"],
                s=45,
                alpha=0.9,
                label=policy,
                edgecolors="white",
                linewidths=0.4,
            )

            for _, row in part.iterrows():
                if pd.notna(row["cost_mean"]) and pd.notna(row["latency_mean"]):
                    load = LOAD_LABELS[LOADS.index(row["load_level"])]
                    ax.annotate(
                        load,
                        (row["cost_mean"], row["latency_mean"]),
                        xytext=(3, 3),
                        textcoords="offset points",
                        fontsize=6.5,
                    )

        ax.set_yscale("log")
        ax.set_title(condition.title())
        ax.set_xlabel("Mean cost per request")
        ax.grid(True, which="major", linestyle=":", linewidth=0.6)
        ax.set_axisbelow(True)

    axes[0].set_ylabel("Mean latency (ms, log scale)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center",
        bbox_to_anchor=(0.5, 1.06), ncol=4, frameon=False,
    )
    fig.suptitle("Latency–cost trade-off across policies and workloads", y=1.14)
    fig.tight_layout()
    save_figure(fig, "03_latency_cost_tradeoff")


def plot_latency_degradation(df):
    """Latency ratio relative to each policy's own low-load latency."""
    fig, axes = plt.subplots(
        1, 3, figsize=(10.5, 3.5), sharey=True
    )

    for ax, condition in zip(axes, CONDITIONS):
        subset = df[df["network_condition"] == condition]

        for policy in POLICIES:
            part = (
                subset[subset["policy"] == policy]
                .set_index("load_level")
                .reindex(LOADS)
            )

            baseline = part.loc["low_load", "latency_mean"]

            if pd.isna(baseline) or baseline <= 0:
                continue

            ratio = part["latency_mean"] / baseline

            ax.plot(
                range(len(LOADS)),
                ratio,
                color=STYLE[policy]["color"],
                marker=STYLE[policy]["marker"],
                linewidth=1.7,
                markersize=4.5,
                label=policy,
            )

        ax.axhline(1, color="black", linestyle="--", linewidth=0.8)
        ax.set_yscale("log")
        ax.set_xticks(range(len(LOADS)))
        ax.set_xticklabels(LOAD_LABELS)
        ax.set_title(condition.title())
        ax.set_xlabel("Workload intensity")
        ax.grid(True, which="major", linestyle=":", linewidth=0.6)
        ax.set_axisbelow(True)

    axes[0].set_ylabel("Latency / low-load latency (×)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center",
        bbox_to_anchor=(0.5, 1.06), ncol=4, frameon=False,
    )
    fig.suptitle("Latency degradation as workload increases", y=1.14)
    fig.tight_layout()
    save_figure(fig, "04_latency_degradation")


def plot_deadline_latency_tradeoff(df):
    """Each point represents one policy and workload combination."""
    fig, axes = plt.subplots(
        1, 3, figsize=(11, 3.8), sharey=True
    )

    for ax, condition in zip(axes, CONDITIONS):
        subset = df[df["network_condition"] == condition]

        for policy in POLICIES:
            part = subset[subset["policy"] == policy]

            x = part["latency_mean"]
            y = 100 * part["violations_mean"]

            ax.scatter(
                x, y,
                color=STYLE[policy]["color"],
                marker=STYLE[policy]["marker"],
                s=45,
                label=policy,
                edgecolors="white",
                linewidths=0.4,
            )

            for (_, row), latency, violation in zip(part.iterrows(), x, y):
                if pd.notna(latency) and pd.notna(violation):
                    load = LOAD_LABELS[LOADS.index(row["load_level"])]
                    ax.annotate(
                        load, (latency, violation),
                        xytext=(3, 3), textcoords="offset points",
                        fontsize=6.5,
                    )

        ax.set_xscale("log")
        ax.set_title(condition.title())
        ax.set_xlabel("Mean latency (ms, log scale)")
        ax.grid(True, which="major", linestyle=":", linewidth=0.6)
        ax.set_axisbelow(True)

    axes[0].set_ylabel("Deadline violations (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center",
        bbox_to_anchor=(0.5, 1.06), ncol=4, frameon=False,
    )
    fig.suptitle("Deadline reliability versus latency", y=1.14)
    fig.tight_layout()
    save_figure(fig, "05_deadline_latency_tradeoff")


def plot_latency_variability(df, run_df):
    """Show mean latency and seed-level observations when available."""
    fig, axes = plt.subplots(
        1, 3, figsize=(11, 4), sharey=True
    )

    for ax, condition in zip(axes, CONDITIONS):
        subset = df[df["network_condition"] == condition]

        for policy in POLICIES:
            part = (
                subset[subset["policy"] == policy]
                .set_index("load_level")
                .reindex(LOADS)
            )

            x = np.arange(len(LOADS))
            mean = part["latency_mean"].to_numpy(dtype=float)
            std = part["latency_std"].to_numpy(dtype=float)

            ax.errorbar(
                x, mean, yerr=std,
                color=STYLE[policy]["color"],
                marker=STYLE[policy]["marker"],
                linewidth=1.5, markersize=4,
                capsize=2, label=policy,
            )

            # Show individual evaluation runs when supplied.
            if not run_df.empty and "latency" in run_df.columns:
                raw = run_df[
                    (run_df["network_condition"] == condition)
                    & (run_df["policy"] == policy)
                ]

                for i, load in enumerate(LOADS):
                    values = raw.loc[
                        raw["load_level"] == load, "latency"
                    ].dropna().to_numpy()

                    if len(values):
                        offsets = np.linspace(-0.08, 0.08, len(values))
                        ax.scatter(
                            i + offsets, values,
                            color=STYLE[policy]["color"],
                            s=10, alpha=0.35, zorder=2,
                        )

        ax.set_xticks(range(len(LOADS)))
        ax.set_xticklabels(LOAD_LABELS)
        ax.set_yscale("log")
        ax.set_title(condition.title())
        ax.set_xlabel("Workload intensity")
        ax.grid(True, which="major", linestyle=":", linewidth=0.6)
        ax.set_axisbelow(True)

    axes[0].set_ylabel("Latency (ms, log scale)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center",
        bbox_to_anchor=(0.5, 1.06), ncol=4, frameon=False,
    )
    fig.suptitle(
        "Latency variability across evaluation runs", y=1.14
    )
    fig.tight_layout()
    save_figure(fig, "06_latency_variability")


def plot_utilization(df):
    """Compare mean utilization as offered workload increases."""
    fig, axes = plt.subplots(
        1, 3, figsize=(10.5, 3.5), sharey=True
    )

    for ax, condition in zip(axes, CONDITIONS):
        subset = df[df["network_condition"] == condition]

        for policy in POLICIES:
            part = (
                subset[subset["policy"] == policy]
                .set_index("load_level")
                .reindex(LOADS)
            )

            ax.plot(
                range(len(LOADS)),
                100 * part["utilization_mean"],
                color=STYLE[policy]["color"],
                marker=STYLE[policy]["marker"],
                linewidth=1.7,
                markersize=4.5,
                label=policy,
            )

        ax.set_xticks(range(len(LOADS)))
        ax.set_xticklabels(LOAD_LABELS)
        ax.set_title(condition.title())
        ax.set_xlabel("Workload intensity")
        ax.grid(axis="y", linestyle=":", linewidth=0.6)
        ax.set_axisbelow(True)

    axes[0].set_ylabel("Mean node utilization (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center",
        bbox_to_anchor=(0.5, 1.06), ncol=4, frameon=False,
    )
    fig.suptitle("Resource utilization under increasing workload", y=1.14)
    fig.tight_layout()
    save_figure(fig, "07_resource_utilization")


def plot_paired_latency_difference(run_df):
    """Paired seed-level difference: XGBoost latency minus Greedy latency."""
    required = {
        "network_condition", "load_level", "policy", "seed", "latency"
    }

    if run_df.empty or not required.issubset(run_df.columns):
        print(
            "Skipping paired latency plot: run-level data must contain "
            "network_condition, load_level, policy, seed, and latency."
        )
        return

    fig, axes = plt.subplots(
        1, 3, figsize=(10.5, 3.5), sharey=True
    )

    for ax, condition in zip(axes, CONDITIONS):
        for i, load in enumerate(LOADS):
            subset = run_df[
                (run_df["network_condition"] == condition)
                & (run_df["load_level"] == load)
            ]

            pivot = subset.pivot_table(
                index="seed", columns="policy", values="latency"
            )

            if not {"Greedy", "XGBoost"}.issubset(pivot.columns):
                continue

            paired = (
                pivot[["XGBoost", "Greedy"]]
                .dropna()
            )
            differences = (
                paired["XGBoost"] - paired["Greedy"]
            ).to_numpy()

            if len(differences) == 0:
                continue

            offsets = np.linspace(-0.08, 0.08, len(differences))
            ax.scatter(
                i + offsets, differences,
                color="#555555", s=20, alpha=0.7,
            )
            ax.plot(
                [i - 0.12, i + 0.12],
                [differences.mean(), differences.mean()],
                color="#CC79A7", linewidth=2,
            )

        ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
        ax.set_xticks(range(len(LOADS)))
        ax.set_xticklabels(LOAD_LABELS)
        ax.set_title(condition.title())
        ax.set_xlabel("Workload intensity")
        ax.grid(axis="y", linestyle=":", linewidth=0.6)
        ax.set_axisbelow(True)

    axes[0].set_ylabel("Paired latency difference (ms)\nXGBoost − Greedy")
    fig.suptitle(
        "Seed-matched latency differences between XGBoost and Greedy",
        y=1.04,
    )
    fig.tight_layout()
    save_figure(fig, "08_paired_latency_difference")


def plot_node_selection_distribution(df, run_df):
    """Plot placement proportions if selection counts are included in results."""
    if run_df.empty:
        print(
            "Skipping node-selection plot: node-selection counts are not "
            "currently included in the experiment results."
        )
        return

    if "node_selections" not in run_df.columns:
        print(
            "Skipping node-selection plot: add a 'node_selections' "
            "dictionary to each run's result row."
        )
        return

    records = []

    for _, row in run_df.iterrows():
        selections = row["node_selections"]

        if not isinstance(selections, dict):
            continue

        total = sum(selections.values())

        if total <= 0:
            continue

        for node, count in selections.items():
            records.append({
                "network_condition": row["network_condition"],
                "load_level": row["load_level"],
                "policy": row["policy"],
                "node": node,
                "proportion": count / total,
            })

    if not records:
        print("Skipping node-selection plot: no valid selection counts found.")
        return

    selections_df = pd.DataFrame(records)
    nodes = sorted(selections_df["node"].unique())

    fig, axes = plt.subplots(
        len(POLICIES), 3,
        figsize=(11, 2.2 * len(POLICIES)),
        sharex=True,
    )

    for row_index, policy in enumerate(POLICIES):
        for col_index, condition in enumerate(CONDITIONS):
            ax = axes[row_index, col_index]
            subset = selections_df[
                (selections_df["policy"] == policy)
                & (selections_df["network_condition"] == condition)
            ]

            pivot = (
                subset.groupby(["load_level", "node"])["proportion"]
                .mean()
                .unstack(fill_value=0)
                .reindex(index=LOADS, columns=nodes, fill_value=0)
            )

            pivot.plot(
                kind="bar",
                stacked=True,
                ax=ax,
                width=0.75,
                legend=False,
                colormap="tab20",
            )

            ax.set_ylim(0, 1)
            ax.set_title(condition.title() if row_index == 0 else "")
            ax.set_xlabel("Workload" if row_index == len(POLICIES) - 1 else "")
            ax.set_ylabel(f"{policy}\nPlacement share" if col_index == 0 else "")
            ax.set_xticklabels(LOAD_LABELS, rotation=0)
            ax.grid(axis="y", linestyle=":", linewidth=0.5)
            ax.set_axisbelow(True)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center",
        bbox_to_anchor=(0.5, 1.01), ncol=min(len(nodes), 5),
        frameon=False,
    )
    fig.suptitle("Node-placement distribution by policy", y=1.04)
    fig.tight_layout()
    save_figure(fig, "09_node_selection_distribution")