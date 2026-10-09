from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


PLOT_DIR = Path(__file__).resolve().parent

CONDITIONS = ["stable", "moderate", "high"]
LOADS = ["low_load", "medium_load", "high_load", "very_high_load"]
LOAD_LABELS = ["Low", "Medium", "High", "Very High"]

POLICIES = ["Always Edge", "Always Cloud", "Greedy", "XGBoost"]
COMPARE_POLICIES = ["Greedy", "XGBoost"]

STYLE = {
    "Always Edge": {"color": "blue", "marker": "o"},
    "Always Cloud": {"color": "red", "marker": "s"},
    "Greedy": {"color": "green", "marker": "^"},
    "XGBoost": {"color": "orange", "marker": "d"},
}


def plot_all_results(summary_rows):
    df = pd.DataFrame(summary_rows)

    if df.empty:
        print("No results available for plotting.")
        return

    plot_violations(df)
    plot_latency(df)
    plot_greedy_vs_xgboost(df)


def plot_violations(df):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)

    for ax, condition in zip(axes, CONDITIONS):
        cond_df = df[df["network_condition"] == condition]

        for policy in POLICIES:
            policy_df = (
                cond_df[cond_df["policy"] == policy]
                .set_index("load_level")
                .reindex(LOADS)
            )

            # Assumes violations_mean is a fraction between 0 and 1.
            y = policy_df["violations_mean"] * 100

            ax.plot(
                LOAD_LABELS,
                y,
                label=policy,
                color=STYLE[policy]["color"],
                marker=STYLE[policy]["marker"],
                linewidth=2,
                markersize=6,
            )

        ax.set_title(condition.title())
        ax.set_xlabel("Load Level")
        ax.grid(True, linestyle="--", linewidth=0.3)

    axes[0].set_ylabel("Deadline Violations (%)")
    axes[0].legend(loc="upper left", fontsize=9)

    fig.suptitle(
        "Deadline violations by policy, load level, and network condition",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "violations_plot.png", dpi=150)
    plt.close(fig)


def plot_latency(df):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)

    for ax, condition in zip(axes, CONDITIONS):
        cond_df = df[df["network_condition"] == condition]

        for policy in POLICIES:
            policy_df = (
                cond_df[cond_df["policy"] == policy]
                .set_index("load_level")
                .reindex(LOADS)
            )

            ax.plot(
                LOAD_LABELS,
                policy_df["latency_mean"],
                label=policy,
                color=STYLE[policy]["color"],
                marker=STYLE[policy]["marker"],
                linewidth=2,
                markersize=6,
            )

        ax.set_yscale("log")
        ax.set_title(condition.title())
        ax.set_xlabel("Load Level")
        ax.grid(True, which="both", linestyle="--", linewidth=0.3)

    axes[0].set_ylabel("Average Latency (ms, log scale)")
    axes[0].legend(loc="upper left", fontsize=9)

    fig.suptitle(
        "Latency by policy, load level, and network condition",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "latency_plot.png", dpi=150)
    plt.close(fig)


def plot_greedy_vs_xgboost(df):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)

    for ax, condition in zip(axes, CONDITIONS):
        cond_df = df[df["network_condition"] == condition]

        for policy in COMPARE_POLICIES:
            policy_df = (
                cond_df[cond_df["policy"] == policy]
                .set_index("load_level")
                .reindex(LOADS)
            )

            ax.errorbar(
                LOAD_LABELS,
                policy_df["latency_mean"],
                yerr=policy_df["latency_std"],
                label=policy,
                color=STYLE[policy]["color"],
                marker=STYLE[policy]["marker"],
                linewidth=2,
                markersize=6,
                capsize=3,
            )

        ax.set_title(condition.title())
        ax.set_xlabel("Load Level")
        ax.grid(True, linestyle="--", linewidth=0.3)

    axes[0].set_ylabel("Average Latency (ms)")
    axes[0].legend(loc="upper left", fontsize=9)

    fig.suptitle("Greedy vs XGBoost latency", fontsize=13)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "greedy_vs_xgboost.png", dpi=150)
    plt.close(fig)

    print(f"Plots saved to {PLOT_DIR}")
