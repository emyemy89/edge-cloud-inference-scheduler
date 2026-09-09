"""
Simulation environment
"""
from dataclasses import dataclass
from nodes.node import Node
from simulation.workload import InferenceRequest

@dataclass
class ActiveRequest:
    request: InferenceRequest
    node: Node
    finish_time: float

class SimulationEnvironment:

    def __init__(self, nodes: list[Node]):
        self.nodes = nodes
        self.current_time = 0
        self.active_requests: list[ActiveRequest] = []

    def reset(self):
        """
        Reset node loads
        """
        self.current_time = 0
        self.active_requests.clear()
        for node in self.nodes:
            node.current_load = 0.0

    def execute(self, node: Node, request: InferenceRequest,) -> float:
        """
        Execute a request on a node.
        Returns estimated latency in milliseconds.
        """
        if not node.can_handle(request.required_compute):
            raise RuntimeError(
                f"{node.name} cannot handle request {request.request_id}"
            )
        latency = node.estimated_latency(request.required_compute)
        node.current_load += request.required_compute
        return latency
