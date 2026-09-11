"""
Node class

Provide logic for creating Node instances used in Cloud and Edge simulations.
"""
from dataclasses import dataclass


@dataclass
class Node:
    name: str
    compute_capacity: float
    network_latency: float
    const_per_request: float
    current_load: float = 0.0

    def can_handle(self, required_compute: float):
        """
        Assess whether this Node can handle compute requests or not.
        """
        return self.current_load + required_compute <= self.compute_capacity

    def estimated_latency(self, required_compute: float):
        """
        Calculate the estimated latency of the Node
        """
        compute_time = required_compute / self.compute_capacity
        return self.network_latency + compute_time
