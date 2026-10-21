"""
Simple experiment for running the baseline
"""

from collections import defaultdict


from simulation.workload import generate_requests
from simulation.runner import run_policy

from scheduler.baselines import always_edge_baseline, always_cloud_baseline, greedy_baseline
from evaluation.results import aggregate_results, print_results, metrics_to_dict


def main():
    policies = {
        "Always Edge": always_edge_baseline,
        "Always Cloud": always_cloud_baseline,
        "Greedy": greedy_baseline,
    }
    # Test for different loads (50 low, 5 very high load)
    workload_scenarios = {"low_load": 50, "medium_load": 20, "high_load": 10, "very_high_load": 5}
    network_scenarios = ["stable", "moderate", "high"]
    evaluation_seeds = [1, 2, 3, 4, 5]
    results = defaultdict(list)
    
    for network_scenario in network_scenarios:
        print(f"\n==============================")
        print(f"NETWORK: {network_scenario.upper()}")
        print(f"==============================")

        for scenario_name, arrival_interval in workload_scenarios.items():
            print(f"\n=== {scenario_name} ===")
            for seed in evaluation_seeds:
                requests = generate_requests(100, seed=seed)
                for policy_name, policy in policies.items():
                    metrics = run_policy(policy, requests, arrival_interval, network_scenario, seed=seed,)
                    print_results(policy_name, metrics, seed)
                    key = (network_scenario, scenario_name, policy_name)
                    results[key].append(metrics_to_dict(metrics))
    aggregate_results(results)

if __name__ == "__main__":
    main()