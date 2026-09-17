"""
The ML-Scheduler simulation
"""
import random
import pandas as pd

from nodes.node import Node
from simulation.workload import generate_requests
from simulation.environment import SimulationEnvironment
from simulation.network import update_network_conditions, NETWORK_UPDATE_INTERVAL
from evaluation.metrics import Metrics

from ml.dataset import get_state_features, generate_training_data
from ml.xgboost_scheduler import predict_node, train_model, evaluate_model

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
             cost_per_request=0.05 ),
    ]

# The ML Model
nodes = create_nodes()
requests = generate_requests(1000)
dataset = generate_training_data(nodes, requests)
model, X_test, y_test = train_model(dataset)

def xgboost_policy(nodes, request):
    return predict_node(model, nodes, request)

# Simulation
def run_ml_policy(requests, arrival_interval, network_scenario, seed=42):
    nodes = create_nodes()
    environment = SimulationEnvironment(nodes)
    metrics = Metrics()

    rng = random.Random(seed)

    for i, request in enumerate(requests):
        if i % NETWORK_UPDATE_INTERVAL == 0:
            update_network_conditions(nodes, network_scenario, rng)
        environment.release_finished_requests()
        selected_node = xgboost_policy(nodes, request)
        latency = environment.execute(selected_node, request)
        metrics.record(
            request_id=request.request_id,
            node_name=selected_node.name,
            latency=latency,
            cost=selected_node.cost_per_request,
            utilization=selected_node.utilization(),
            deadline=request.deadline,
        )
        environment.advance_time(arrival_interval)
    return metrics



# Boilerplate and main function, to be cleaned later
def print_results(metrics):
    print(f"Average latency: {metrics.average_latency():.2f} ms")
    print(f"Total cost: {metrics.total_cost():.2f}")
    print(f"Average utilization: {metrics.average_utilization():.2%}")
    print(f"Deadline violation rate: {metrics.deadline_violation_rate():.2%}")
    print(f"Node selections: {metrics.node_selection_counts()}")


def main():
    print("\n---- XGBoost classification ----")
    evaluate_model(model, X_test, y_test)
    workload_scenarios = {"low_load": 50, "medium_load": 20, "high_load": 10, "very_high_load": 5}
    network_scenarios = ["stable", "moderate", "high"]
    requests = generate_requests(100)

    for network_scenario in network_scenarios:
        print(f"\n==============================")
        print(f"NETWORK: {network_scenario.upper()}")
        print(f"==============================")

        for scenario_name, arrival_interval in workload_scenarios.items():
            print(f"\n=== {scenario_name} ===")

            metrics = run_ml_policy(
                requests,
                arrival_interval,
                network_scenario,
                seed=42,
            )

            print_results(metrics)


if __name__ == "__main__":
    main()
