import statistics
import csv
from collections import defaultdict

from nodes.node import create_nodes
from scheduler.baselines import (
    always_edge_baseline,
    always_cloud_baseline,
    greedy_baseline,
)
from simulation.workload import generate_requests
from simulation.runner import run_policy
from ml.xgboost_scheduler import train_model, xgboost_policy
from ml.dataset import generate_training_data
from evaluation.results import aggregate_results, metrics_to_dict


evaluation_seeds = [1, 2, 3, 4, 5]
workload_scenarios = {"low_load": 50, "medium_load": 20, "high_load": 10, "very_high_load": 5}
network_scenarios = ["stable", "moderate", "high"]

RESULTS_CSV_PATH = "experiments/results.csv"


def aggregate_results(results):
    rows = []

    for key, runs in results.items():
        network_scenario, scenario_name, policy_name = key

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
nodes = create_nodes()
training_requests = generate_requests(10000)
dataset = generate_training_data(nodes, training_requests)
model, X_test, y_test = train_model(dataset)

def ml_policy(nodes, request):
    return xgboost_policy(model, nodes, request)

def main():

    policies = {
        "Always Edge": always_edge_baseline,
        "Always Cloud": always_cloud_baseline,
        "Greedy": greedy_baseline,
        "XGBoost": ml_policy,
    }

    results = defaultdict(list)

    for network_scenario in network_scenarios:
        for scenario_name, arrival_interval in workload_scenarios.items():
            for seed in evaluation_seeds:

                requests = generate_requests(1000, seed=seed)

                for policy_name, policy in policies.items():
                    metrics = run_policy(policy, requests, arrival_interval, network_scenario, seed=seed,)
                    results[(network_scenario, scenario_name, policy_name)
                    ].append(metrics_to_dict(metrics))

    print("\n\n==============================")
    print("SCHEDULER COMPARISON")
    print("==============================")

    aggregate_results(results)


if __name__ == "__main__":
    main()