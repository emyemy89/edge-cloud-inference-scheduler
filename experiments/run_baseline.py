"""
Simple experiment for running the baseline
"""
import random

from nodes.node import Node
from simulation.workload import generate_requests
from simulation.environment import SimulationEnvironment
from simulation.network import (update_network_conditions, NETWORK_UPDATE_INTERVAL)
from scheduler.baselines import always_edge_baseline, always_cloud_baseline, greedy_baseline
from evaluation.metrics import Metrics
from ml.dataset import generate_training_data


def create_nodes() -> list[Node]:
    return [
        Node(name="edge_1",
             compute_capacity=10,
             network_latency=5,
             cost_per_request=0.01),
        Node(name="edge_2",
             compute_capacity=20,
             network_latency=10,
             cost_per_request=0.015, ),
        Node(name="cloud",
             compute_capacity=100,
             network_latency=50,
             cost_per_request=0.05, ),
    ]

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


def print_results(policy_name: str, metrics: Metrics):
    print(f"\n--- {policy_name} ---")
    print(f"Average latency: {metrics.average_latency():.2f} ms")
    print(f"Total cost: {metrics.total_cost():.2f}")
    print(f"Average utilization: " f"{metrics.average_utilization():.2%}")
    print(f"Deadline violation rate: " f"{metrics.deadline_violation_rate():.2%}")
    print(f"Node selections: {metrics.node_selection_counts()}")


def main():
    requests = generate_requests(100)
    policies = {
        "Always Edge": always_edge_baseline,
        "Always Cloud": always_cloud_baseline,
        "Greedy": greedy_baseline,
    }
    # Test for different loads (50 low, 5 very high load)
    workload_scenarios = {"low_load": 50, "medium_load": 20, "high_load": 10, "very_high_load": 5}
    network_scenarios = ["stable", "moderate", "high"]
    
    for network_scenario in network_scenarios:
        print(f"\n==============================")
        print(f"NETWORK: {network_scenario.upper()}")
        print(f"==============================")

        for scenario_name, arrival_interval in workload_scenarios.items():
            print(f"\n=== {scenario_name} ===")
            for policy_name, policy in policies.items():
                metrics = run_policy(policy, requests, arrival_interval, network_scenario, seed=42)
                print_results(policy_name, metrics)
    # Dummy test for generating dataset
    nodes = create_nodes()
    requests = generate_requests(1000)
    dataset = generate_training_data(nodes, requests)
    print(f"Dataset shape: {dataset.shape}")
    print("\nColumns:")
    print(dataset.columns.tolist())
    print("\nTarget distribution:")
    print(dataset["target"].value_counts())
    print("\nFirst 5 rows:")
    print(dataset.head())

if __name__ == "__main__":
    main()