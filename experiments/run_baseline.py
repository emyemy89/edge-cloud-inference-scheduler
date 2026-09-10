"""
Simple experiment for running the baseline
"""
from nodes.node import Node
from simulation.workload import generate_requests
from simulation.environment import SimulationEnvironment
from scheduler.baselines import greedy_baseline


def main():

    nodes = [
        Node(name="edge_1",
            compute_capacity=10,
            network_latency=5,
            cost_per_request=0.01),
        Node(name="edge_2",
            compute_capacity=20,
            network_latency=10,
            cost_per_request=0.015,),
        Node(name="cloud",
            compute_capacity=100,
            network_latency=50,
            cost_per_request=0.05,),
    ]

    environment = SimulationEnvironment(nodes)
    requests = generate_requests(100)

    environment.reset()

    for request in requests:
        selected_node = greedy_baseline(nodes, request)
        latency = environment.execute(selected_node, request,)

        print(
            f"Request {request.request_id}: "
            f"{selected_node.name}, "
            f"latency={latency:.2f} ms"
        )


if __name__ == "__main__":
    main()