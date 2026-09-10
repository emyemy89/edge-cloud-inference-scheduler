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

    def release_finished_requests(self):
        """
        Release the finished requests
        """
        remaining_requests = []
        for active in self.active_requests:
            if active.finish_time <= self.current_time:
                active.node.remove_load(active.request.required_compute)
            else:
                remaining_requests.append(active)
            self.active_requests = remaining_requests


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
        node.add_load(request.required_compute)
        active_request = ActiveRequest(request=request, node=node, finish_time=self.current_time + latency)
        self.active_requests.append(active_request)
        return latency

    def advance_time(self, amount:float):
        """
        Advance the simulation time
        """
        self.current_time += amount
        self.release_finished_requests()
