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
    edge_nodes = [node for node in nodes
                  if "edge" in node.name and node.can_handle(request.required_compute)]
    if not edge_nodes:
        raise RuntimeError("No available edge node")
    return edge_nodes[0]

def always_cloud(nodes: list[Node], request: InferenceRequest) -> Node:
    """
    Always pick cloud
    """
    cloud = next(node for node in nodes if node.name == "cloud")
    if not cloud.can_handle(request.required_compute):
        raise RuntimeError("Cloud can't handle request")
    return cloud


def lowest_latency(nodes: list[Node], request: InferenceRequest) -> Node:
    """
    Pick the lowest latency node
    """
    return min(
        nodes,
        key=lambda node: node.estimated_latency(request.required_compute),
    )