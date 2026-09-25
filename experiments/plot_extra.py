import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("experiments/results.csv")

conditions = ["stable", "moderate", "high"]
loads = ["low_load", "medium_load", "high_load", "very_high_load"]
load_labels = ["Low", "Medium", "High", "Very High"]

style = {
    "Always Edge": {"color": "blue", "marker": "o"},
    "Always Cloud": {"color": "red", "marker": "s"},
    "Greedy": {"color": "green", "marker": "^"},
    "XGBoost": {"color": "orange", "marker": "d"},
}


# --- Plot 1: deadline violations, all four policies -----------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)

for ax, condition in zip(axes, conditions):
    cond_df = df[df["network_condition"] == condition]
    for policy in style:
        policy_df = cond_df[cond_df["policy"] == policy].set_index("load_level").reindex(loads)
        y = policy_df["violations_mean"]
        ax.plot(
            load_labels, y,
            label=policy,
            color=style[policy]["color"],
            marker=style[policy]["marker"],
            linewidth=2,
            markersize=6,
        )
    ax.set_title(condition.title())
    ax.set_xlabel("Load Level")
    ax.grid(True, linestyle="--", linewidth=0.3)

axes[0].set_ylabel("Deadline Violations (%)")
axes[0].legend(loc="upper left", fontsize=9)

fig.suptitle("Deadline violations by policy, load level, and network condition", fontsize=13)
fig.tight_layout()
fig.savefig("experiments/violations_plot.png", dpi=150)
print("Saved to experiments/violations_plot.png")


# --- Plot 2: Greedy vs XGBoost only, linear scale, zoomed in --------------
compare_policies = ["Greedy", "XGBoost"]

fig2, axes2 = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)

for ax, condition in zip(axes2, conditions):
    cond_df = df[df["network_condition"] == condition]
    for policy in compare_policies:
        policy_df = cond_df[cond_df["policy"] == policy].set_index("load_level").reindex(loads)
        y = policy_df["latency_mean"]
        yerr = policy_df["latency_std"]
        ax.errorbar(
            load_labels, y, yerr=yerr,
            label=policy,
            color=style[policy]["color"],
            marker=style[policy]["marker"],
            linewidth=2,
            markersize=6,
            capsize=3,
        )
    ax.set_title(condition.title())
    ax.set_xlabel("Load Level")
    ax.grid(True, linestyle="--", linewidth=0.3)
    # no log scale here - values are close together, linear makes the gap visible

axes2[0].set_ylabel("Average Latency (ms)")
axes2[0].legend(loc="upper left", fontsize=9)

fig2.suptitle("Greedy vs XGBoost latency (zoomed)", fontsize=13)
fig2.tight_layout()
fig2.savefig("experiments/greedy_vs_xgboost.png", dpi=150)
print("Saved to experiments/greedy_vs_xgboost.png")
