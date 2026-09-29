import random

NETWORK_UPDATE_INTERVAL = 10


def update_network_conditions(nodes, scenario, rng):
    """
    Update node network latencies according to the selected scenario.
    """
    if scenario == "stable":
        return
    if scenario == "moderate":
        min_factor = 0.8
        max_factor = 1.2
    elif scenario == "high":
        min_factor = 0.5
        max_factor = 2.0
    else:
        raise ValueError(f"Unknown network scenario: {scenario}")

    for node in nodes:
        node.network_latency = node.base_network_latency * rng.uniform(min_factor, max_factor)
