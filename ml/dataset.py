import pandas as pd

from nodes.node import Node
from simulation.workload import InferenceRequest

def get_state_features(nodes: list[Node], request: InferenceRequest) -> dict:
    """
    Extract the information available to a scheduler before placement.
    """
    features = {"required_compute": request.required_compute, "deadline": request.deadline}
    for node in nodes:
        prefix = node.name
        features[f"{prefix}_compute_capacity"] = node.compute_capacity
        features[f"{prefix}_network_latency"] = node.network_latency
        features[f"{prefix}_utilization"] = node.utilization()
        features[f"{prefix}_cost"] = node.cost_per_request
    return features

def generate_training_data(nodes: list[Node], requests: list[InferenceRequest]) -> pd.DataFrame:
    """
    Generate one training example for every request.
    Each example contains the system state and the best node according
    to the simulated latency.
    """
    rows = []
    for request in requests:
        state = get_state_features(nodes, request)
        node_latencies = {}
        for node in nodes:
            if node.can_handle(request.required_compute):
                node_latencies[node.name] = node.estimated_latency(
                    request.required_compute
                )
        if not node_latencies:
            continue
        best_node = min(node_latencies, key=node_latencies.get)
        row = state.copy()
        row["target"] = best_node
        rows.append(row)
    return pd.DataFrame(rows)
