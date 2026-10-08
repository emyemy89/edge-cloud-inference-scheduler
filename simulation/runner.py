import random

from nodes.node import create_nodes
from simulation.environment import SimulationEnvironment
from simulation.network import (update_network_conditions, NETWORK_UPDATE_INTERVAL)
from evaluation.metrics import Metrics


def run_policy(policy, requests, arrival_interval, network_scenario, seed=42):
    # We generate new nodes for every baseline
     nodes = create_nodes()
     environment = SimulationEnvironment(nodes)
     metrics = Metrics()
     environment.reset()
     rng = random.Random(seed)
     # Request loop
     for i, request in enumerate(requests):
         # Change network conditions every 'NET_UPDT' time
         if i % NETWORK_UPDATE_INTERVAL == 0:
             update_network_conditions(nodes, network_scenario, rng)
         environment.release_finished_requests()
         selected_node = policy(nodes, request)
         latency = environment.execute(selected_node, request)
         metrics.record(request_id=request.request_id,
                        node_name=selected_node.name,
                        latency=latency,
                        cost=selected_node.cost_per_request,
                        utilization=selected_node.utilization(),
                        deadline=request.deadline)
         environment.advance_time(arrival_interval)
     return metrics