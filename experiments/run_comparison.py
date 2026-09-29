import statistics
from collections import defaultdict

from simulation.workload import generate_requests
from experiments.run_baseline import run_policy, create_nodes
from scheduler.baselines import (
    always_edge_baseline,
    always_cloud_baseline,
    greedy_baseline,
)
from experiments.run_ml import run_ml_policy


evaluation_seeds = [1, 2, 3, 4, 5]
workload_scenarios = {"low_load": 50, "medium_load": 20, "high_load": 10, "very_high_load": 5}
network_scenarios = ["stable", "moderate", "high"]


def aggregate_results(results):
    for key, runs in results.items():
        latencies = [run["latency"] for run in runs]
        costs = [run["cost"] for run in runs]
        utilizations = [run["utilization"] for run in runs]
        violations = [run["violations"] for run in runs]
        print(f"\n{key[0].upper()} / {key[1]} / {key[2]}")
        print(f"Average latency: "f"{statistics.mean(latencies):.2f} ± "f"{statistics.stdev(latencies):.2f} ms")

        print(f"Average cost: "f"{statistics.mean(costs):.2f} ± "f"{statistics.stdev(costs):.2f}")

        print(
            f"Average utilization: "
            f"{statistics.mean(utilizations):.2%} ± "
            f"{statistics.stdev(utilizations):.2%}"
        )

        print(
            f"Average deadline violations: "
            f"{statistics.mean(violations):.2%} ± "
            f"{statistics.stdev(violations):.2%}"
        )


def main():
    policies = {
        "Always Edge": run_policy,
        "Always Cloud": run_policy,
        "Greedy": run_policy,
        "XGBoost": run_ml_policy,
    }

    baseline_policies = {
        "Always Edge": always_edge_baseline,
        "Always Cloud": always_cloud_baseline,
        "Greedy": greedy_baseline,
    }

    results = defaultdict(list)

    for network_scenario in network_scenarios:
        for scenario_name, arrival_interval in workload_scenarios.items():
            for seed in evaluation_seeds:

                requests = generate_requests(100, seed=seed)

                for policy_name, policy in baseline_policies.items():
                    metrics = run_policy(
                        policy,
                        requests,
                        arrival_interval,
                        network_scenario,
                        seed=seed,
                    )

                    results[
                        (network_scenario, scenario_name, policy_name)
                    ].append({
                        "latency": metrics.average_latency(),
                        "cost": metrics.total_cost(),
                        "utilization": metrics.average_utilization(),
                        "violations": metrics.deadline_violation_rate(),
                    })

                metrics = run_ml_policy(
                    requests,
                    arrival_interval,
                    network_scenario,
                    seed=seed,
                )

                results[
                    (network_scenario, scenario_name, "XGBoost")
                ].append({
                    "latency": metrics.average_latency(),
                    "cost": metrics.total_cost(),
                    "utilization": metrics.average_utilization(),
                    "violations": metrics.deadline_violation_rate(),
                })

    print("\n\n==============================")
    print("SCHEDULER COMPARISON")
    print("==============================")

    aggregate_results(results)


if __name__ == "__main__":
    main()