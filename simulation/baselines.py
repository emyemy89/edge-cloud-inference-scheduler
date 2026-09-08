"""
Baselines definition

Logic for: always edge, always cloud and Greedy
"""
from nodes.node import Node
from simulation.workload import InferenceRequest

def always_edge(nodes: list[Node], request: InferenceRequest) -> Node:
    """
    Always pick the first available edge
    """
    edge_nodes = [node for node in nodes if "edge" in node.name]
    return edge_nodes[0]

def always_cloud(nodes: list[Node], request: InferenceRequest) -> Node:
    """
    Always pick cloud
    """
    return next(node for node in nodes if node.name == "cloud")


def lowest_latency(nodes: list[Node], request: InferenceRequest) -> Node:
    """
    Pick the lowest latency node
    """
    return min(
        nodes,
        key=lambda node: node.estimated_latency(request.required_compute),
    )