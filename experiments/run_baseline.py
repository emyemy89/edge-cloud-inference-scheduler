"""
Simple experiment for running the baseline
"""
import random

from collections import defaultdict

from nodes.node import create_nodes

from simulation.workload import generate_requests
from simulation.environment import SimulationEnvironment
from simulation.network import (update_network_conditions, NETWORK_UPDATE_INTERVAL)

from scheduler.baselines import always_edge_baseline, always_cloud_baseline, greedy_baseline
from evaluation.metrics import Metrics
from evaluation.results import aggregate_results, print_results


def run_policy(policy, requests, arrival_interval, network_scenario, seed=42):
    # We generate new nodes for every baseline
     nodes = create_nodes()
     environment = SimulationEnvironment(nodes)
     metrics = Metrics()
     environment.reset()
     rng = random.Random(seed)
     # Request loop
     for i, request in enumerate(requests):
         # Change network conditions every 'NET_UPDT' time
         if i % NETWORK_UPDATE_INTERVAL == 0:
             update_network_conditions(nodes, network_scenario, rng)
         environment.release_finished_requests()
         selected_node = policy(nodes, request)
         latency = environment.execute(selected_node, request)
         metrics.record(request_id=request.request_id,
                        node_name=selected_node.name,
                        latency=latency,
                        cost=selected_node.cost_per_request,
                        utilization=selected_node.utilization(),
                        deadline=request.deadline)
         environment.advance_time(arrival_interval)
     return metrics


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
                    results[key].append({
                        "latency": metrics.average_latency(),
                        "cost": metrics.total_cost(),
                        "utilization": metrics.average_utilization(),
                        "violations": metrics.deadline_violation_rate(),
                    })
    aggregate_results(results)

if __name__ == "__main__":
    main()