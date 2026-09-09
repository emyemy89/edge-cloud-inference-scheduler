from nodes.node import Node
from simulation.workload import InferenceRequest


class SimulationEnvironment:

    def __init__(self, nodes: list[Node]):
        self.nodes = nodes

    def reset(self):
        """
        Reset node loads
        """
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
    