"""
The ML-Scheduler simulation
"""
from collections import defaultdict

from nodes.node import create_nodes
from simulation.workload import generate_requests
from simulation.runner import run_policy
from evaluation.results import aggregate_results, print_results, metrics_to_dict

from ml.dataset import generate_training_data
from ml.xgboost_scheduler import train_model, evaluate_model, xgboost_policy

# The ML Model
nodes = create_nodes()
requests = generate_requests(10000)
dataset = generate_training_data(nodes, requests)
model, X_test, y_test = train_model(dataset)


def policy(nodes, request):
    return xgboost_policy(model, nodes, request)

def main():
    print("\n---- XGBoost classification ----")
    evaluate_model(model, X_test, y_test)
    workload_scenarios = {"low_load": 50, "medium_load": 20, "high_load": 10, "very_high_load": 5}
    network_scenarios = ["stable", "moderate", "high"]
    evaluation_seeds = [1, 2, 3, 4, 5]
    results = defaultdict(list) # Store the results

    for network_scenario in network_scenarios:
        print(f"\n==============================")
        print(f"NETWORK: {network_scenario.upper()}")
        print(f"==============================")

        for scenario_name, arrival_interval in workload_scenarios.items():
            print(f"\n=== {scenario_name} ===")

            for seed in evaluation_seeds:
                requests = generate_requests(100, seed=seed)

                metrics = run_policy(policy, requests, arrival_interval, network_scenario, seed=seed,)
                print_results(None, metrics, seed)
                # Store the results
                key = (network_scenario, scenario_name)
                results[key].append(metrics_to_dict(metrics))
    aggregate_results(results)


if __name__ == "__main__":
    main()
