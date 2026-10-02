import random
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

def generate_training_data(nodes: list[Node], requests: list[InferenceRequest], seed: int = 42) -> pd.DataFrame:
    """
    Generate one training example for every request.
    Each example contains the system state and the best node according
    to the simulated latency, cost and deadline
    """
    rng = random.Random(seed)
    rows = []
    for request in requests:
        # Choose the type of system state
        state_type = rng.choices(["normal", "congested", "severely_congested"], weights=[0.5, 0.3, 0.2], k=1)[0]
        for node in nodes:
            if "edge" in node.name:
                if state_type == "normal":
                    utilization = rng.uniform(0.0, 0.6)
                elif state_type == "congested":
                    utilization = rng.uniform(0.6, 0.9)
                else:
                    utilization = rng.uniform(0.9, 1.0)
                node.current_load = (utilization * node.compute_capacity)
                node.network_latency = (node.base_network_latency* rng.uniform(0.5, 2.0))
            else:
                # Cloud is generally less utilized.
                node.current_load = (rng.uniform(0.0, 0.3)* node.compute_capacity)
                node.network_latency = (node.base_network_latency * rng.uniform(0.5, 1.5))
        state = get_state_features(nodes, request)
        node_scores = {}
        for node in nodes:
            if not node.can_handle(request.required_compute):
                continue
            latency = node.estimated_latency(request.required_compute)
            cost = node.cost_per_request
            # Penalize missing the deadline
            deadline_penalty = 100 if latency>request.deadline else 0
            score = latency + cost*100 + deadline_penalty
            node_scores[node.name] = score
        if not node_scores:
            continue
        best_node = min(node_scores, key=node_scores.get)
        row = state.copy()
        row["target"] = best_node
        rows.append(row)
    return pd.DataFrame(rows)
