"""
Simple experiment for running the baseline
"""
from nodes.node import Node
from simulation.workload import generate_requests
from simulation.environment import SimulationEnvironment
from scheduler.baselines import greedy_baseline
from evaluation.metrics import Metrics


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
    metrics = Metrics()

    # Request loop
    for request in requests:
        environment.release_finished_requests()
        selected_node = greedy_baseline(nodes, request)
        latency = environment.execute(selected_node, request)
        metrics.record(request_id=request.request_id,
                       node_name=selected_node.name,
                       latency=latency,
                       cost=selected_node.cost_per_request,
                       utilization=selected_node.utilization(),
                       deadline= request.deadline)

        print(
            f"t={environment.current_time:.2f} ms | "
            f"Request {request.request_id}: "
            f"{selected_node.name}, "
            f"latency={latency:.2f} ms"
        )
        # Simulate requests arriving every 20 ms
        environment.advance_time(20)

    print("\n--- Results ---")
    print(f"Average latency: {metrics.average_latency():.2f} ms")
    print(f"Total cost: {metrics.total_cost():.2f}")
    print(f"Average utilization: {metrics.average_utilization():.2%}")
    print(
        f"Deadline violation rate: "
        f"{metrics.deadline_violation_rate():.2%}"
    )

if __name__ == "__main__":
    main()