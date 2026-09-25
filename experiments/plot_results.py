import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("experiments/results.csv")

conditions = ["stable", "moderate", "high"]
loads = ["low_load", "medium_load", "high_load", "very_high_load"]
load_labels = ["Low", "Medium", "High", "Very High"]
policies = ["Always Edge", "Always Cloud", "Greedy", "XGBoost"]

style = {
    "Always Edge": {"color": "blue", "marker": "o"},
    "Always Cloud": {"color": "red", "marker": "s"},
    "Greedy": {"color": "green", "marker": "^"},
    "XGBoost": {"color": "orange", "marker": "d"}
}

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)

for ax, condition in zip(axes, conditions):
    cond_df = df[df["network_condition"] == condition]
    for policy in policies:
        policy_df = cond_df[cond_df["policy"] == policy].set_index("load_level").reindex(loads)
        y = policy_df["latency_mean"]
        ax.plot(
            load_labels, y,
            label=policy,
            color=style[policy]["color"],
            marker=style[policy]["marker"],
            linewidth=2,
            markersize=6,
        )
    ax.set_yscale("log")
    ax.set_title(condition.title())
    ax.set_xlabel("Load Level")
    ax.grid(True, which="both", linestyle="--", linewidth=0.3)

axes[0].set_ylabel("Average Latency (ms, log scale)")
axes[0].legend(loc="upper left", fontsize=9)

fig.suptitle("Latency by policy, load level, and network condition", fontsize=13)
fig.tight_layout()
fig.savefig("experiments/latency_plot.png", dpi=150)
print("Saved to experiments/latency_plot.png")