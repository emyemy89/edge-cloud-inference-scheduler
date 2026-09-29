from dataclasses import dataclass
from collections import Counter

@dataclass
class RequestResults:
    request_id: int
    node_name: str
    latency: float
    cost: float
    utilization: float
    deadline: float
    deadline_met: bool

class Metrics():
    def __init__(self):
        self.results: list[RequestResults] = []
        self.node_counts = Counter()

    def record(self, request_id: int,
        node_name: str,
        latency: float,
        cost: float,
        utilization: float,
        deadline: float
    ):
        self.results.append(RequestResults(request_id, node_name, latency, cost, utilization, deadline, latency<=deadline))
        self.node_counts[node_name] += 1
    def average_latency(self)->float:
        """
        Average latency over all nodes
        """
        if not self.results:
            return 0
        return sum(result.latency for result in self.results)/len(self.results)

    def total_cost(self)->float:
        """
        Total latency over all nodes
        """
        if not self.results:
            return 0
        return sum(result.cost for result in self.results)

    def average_utilization(self)->float:
        """
        Average utilization over all nodes
        """
        if not self.results:
            return 0
        return sum(result.utilization for result in self.results)/len(self.results)

    def deadline_violation_rate(self)->float:
        if not self.results:
            return 0
        violations = sum(not result.deadline_met for result in self.results)
        return violations/len(self.results)

    def node_selection_counts(self) -> dict[str, int]:
        return dict(self.node_counts)
