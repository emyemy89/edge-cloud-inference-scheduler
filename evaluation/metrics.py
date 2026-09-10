from dataclasses import dataclass

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

    def record(self, request_id: int,
        node_name: str,
        latency: float,
        cost: float,
        utilization: float,
        deadline: float
    ):
        self.results.append(RequestResults(request_id, node_name, latency, cost, utilization, deadline, latency<=deadline))

    def average_latency(self)->float:
        if not self.results:
            return 0
        return sum(result.latency for result in self.results)/len(self.results)

    def total_latency(self)->float:
        if not self.results:
            return 0
        return sum(result.letency for result in self.results)

    
