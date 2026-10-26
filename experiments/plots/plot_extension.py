import statistics
from pathlib import Path
import matplotlib.pyplot as plt

def plot_real_world_results(all_seed_results, policies, seeds, nodes):
    """Plot results from the real-world end-to-end experiment."""

    plot_dir = Path(__file__).resolve().parent / "figures"
    plot_dir.mkdir(parents=True, exist_ok=True)

    policy_names = list(policies)
    colors = {
        "Always Edge": "#0072B2",
        "Always Cloud": "#D55E00",
        "Greedy": "#009E73",
        "XGBoost": "#CC79A7",
    }
    markers = {
        "Always Edge": "o",
        "Always Cloud": "s",
        "Greedy": "^",
        "XGBoost": "D",
    }

    # Aggregate request-level measurements into one value per seed.
    metrics = {
        "End-to-end latency": "end_to_end_time_ms",
        "Inference latency": "inference_time_ms",
        "API total time": "api_time_ms",
        "Scheduler time": "scheduler_time_ms",
        "Cost": "cost",
    }

    summary = {policy: {} for policy in policy_names}

    for policy in policy_names:
        for label, metric in metrics.items():
            seed_means = [
                statistics.mean(
                    result[metric]
                    for result in all_seed_results[seed][policy]
                )
                for seed in seeds
            ]

            summary[policy][label] = (
                statistics.mean(seed_means),
                statistics.stdev(seed_means) if len(seed_means) > 1 else 0.0,
            )

        summary[policy]["Deadline violations"] = (
            statistics.mean([
                sum(
                    not result["deadline_met"]
                    for result in all_seed_results[seed][policy]
                ) / len(all_seed_results[seed][policy]) * 100
                for seed in seeds
            ]),
            statistics.stdev([
                sum(
                    not result["deadline_met"]
                    for result in all_seed_results[seed][policy]
                ) / len(all_seed_results[seed][policy]) * 100
                for seed in seeds
            ]) if len(seeds) > 1 else 0.0,
        )

    # --------------------------------------------------
    # Plot 1: End-to-end latency by policy

    fig, ax = plt.subplots(figsize=(7, 4.5))

    for policy in policy_names:
        mean, std = summary[policy]["End-to-end latency"]

        ax.errorbar(
            policy, mean, yerr=std,
            color=colors[policy],
            marker=markers[policy],
            markersize=7,
            capsize=4,
            linewidth=1.5,
        )

    ax.set_ylabel("Mean end-to-end latency (ms)")
    ax.set_title("End-to-end latency by scheduling policy")
    ax.grid(axis="y", linestyle=":", linewidth=0.6)
    ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(plot_dir / "10_realworld_end_to_end_latency.png",
                dpi=300, bbox_inches="tight")
    plt.close(fig)

    # --------------------------------------------------
    # Plot 2: Timing measurements
    # These measurements overlap and must not be stacked.

    timing_metrics = [
        "Scheduler time",
        "Inference latency",
        "API total time",
        "End-to-end latency",
    ]

    fig, ax = plt.subplots(figsize=(9, 5))

    width = 0.18
    x = list(range(len(timing_metrics)))

    for i, policy in enumerate(policy_names):
        means = [summary[policy][metric][0] for metric in timing_metrics]
        stds = [summary[policy][metric][1] for metric in timing_metrics]
        offsets = [position + (i - (len(policy_names) - 1) / 2) * width
                   for position in x]

        ax.bar(
            offsets, means, width,
            yerr=stds,
            capsize=2,
            color=colors[policy],
            label=policy,
            edgecolor="white",
            linewidth=0.5,
        )

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(timing_metrics, rotation=15, ha="right")
    ax.set_ylabel("Mean measured time (ms, log scale)")
    ax.set_title("Timing measurements across scheduling policies")
    ax.grid(axis="y", which="major", linestyle=":", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=2)

    fig.tight_layout()
    fig.savefig(plot_dir / "11_realworld_timing_measurements.png",
                dpi=300, bbox_inches="tight")
    plt.close(fig)

    # --------------------------------------------------
    # Plot 3: Deadline violations

    fig, ax = plt.subplots(figsize=(7, 4.5))

    for policy in policy_names:
        mean, std = summary[policy]["Deadline violations"]

        ax.errorbar(
            policy, mean, yerr=std,
            color=colors[policy],
            marker=markers[policy],
            markersize=7,
            capsize=4,
            linewidth=1.5,
        )

    ax.set_ylabel("Deadline violations (%)")
    ax.set_ylim(bottom=0)
    ax.set_title("Deadline violations by scheduling policy")
    ax.grid(axis="y", linestyle=":", linewidth=0.6)
    ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(plot_dir / "12_realworld_deadline_violations.png",
                dpi=300, bbox_inches="tight")
    plt.close(fig)

    # --------------------------------------------------
    # Plot 4: Node selection distribution

    selection_counts = {
        policy: {node.name: 0 for node in nodes}
        for policy in policy_names
    }

    for seed in seeds:
        for policy in policy_names:
            for result in all_seed_results[seed][policy]:
                selection_counts[policy][result["node"]] += 1

    fig, ax = plt.subplots(figsize=(8, 4.5))

    bottom = [0.0] * len(policy_names)
    node_colors = ["#0072B2", "#009E73", "#D55E00"]

    for i, node in enumerate(nodes):
        proportions = [
            selection_counts[policy][node.name]
            / sum(selection_counts[policy].values())
            * 100
            for policy in policy_names
        ]

        ax.bar(
            policy_names,
            proportions,
            bottom=bottom,
            label=node.name,
            color=node_colors[i % len(node_colors)],
            edgecolor="white",
            linewidth=0.5,
        )

        bottom = [
            previous + current
            for previous, current in zip(bottom, proportions)
        ]

    ax.set_ylabel("Requests assigned (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Node selection distribution by policy")
    ax.legend(title="Execution node", frameon=False)
    ax.grid(axis="y", linestyle=":", linewidth=0.6)
    ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(plot_dir / "13_realworld_node_selection.png",
                dpi=300, bbox_inches="tight")
    plt.close(fig)

    # --------------------------------------------------
    # Plot 5: Cost versus end-to-end latency

    fig, ax = plt.subplots(figsize=(7, 4.5))

    for policy in policy_names:
        latency = summary[policy]["End-to-end latency"][0]
        cost = summary[policy]["Cost"][0]

        ax.scatter(
            latency, cost,
            color=colors[policy],
            marker=markers[policy],
            s=75,
            label=policy,
            zorder=3,
        )

        ax.annotate(
            policy,
            (latency, cost),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )

    ax.set_xlabel("Mean end-to-end latency (ms)")
    ax.set_ylabel("Mean cost per request")
    ax.set_title("Cost–latency trade-off")
    ax.grid(True, linestyle=":", linewidth=0.6)
    ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(plot_dir / "14_realworld_cost_latency.png",
                dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"\nSaved real-world experiment plots to: {plot_dir}")