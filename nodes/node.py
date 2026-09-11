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
    cost_per_request: float
    current_load: float = 0.0

    def can_handle(self, required_compute: float):
        """
        Assess whether this Node can handle compute requests or not.
        """
        return required_compute <= self.compute_capacity

    def estimated_latency(self, required_compute: float):
        """
        Calculate the estimated latency of the Node based on network latency and current load
        """
        projected_load = self.current_load + required_compute
        load_factor = projected_load / self.compute_capacity

        compute_time = required_compute / self.compute_capacity
        queue_delay = max(0.0, load_factor - 1.0)*100
        return self.network_latency + compute_time + queue_delay

    def add_load(self, required_compute: float):
        """
        Add load to the current load
        """
        self.current_load += required_compute

    def remove_load(self, required_compute: float):
        """
        Remove load from current, make sure it is not less than 0
        """
        self.current_load = max(0, int(self.current_load - required_compute))

    def utilization(self):
        """
        Calculate the utilization of this Node
        """
        return min(1.0, self.current_load / self.compute_capacity)



