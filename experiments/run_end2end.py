import time
import statistics
import random
from pathlib import Path

from inference.client import send_inference
from ml.dataset import generate_training_data
from ml.xgboost_scheduler import train_model, predict_node
from nodes.node import Node
from simulation.workload import InferenceRequest, generate_requests
from scheduler.baselines import always_edge_baseline, always_cloud_baseline, greedy_baseline


# --------------------------------------------------
# Configuration

IMAGE_PATH = Path("../inference/dog.png")
NUM_REQUESTS = 50

# Same basic node configuration as the simulator
NODES = [
    Node(name="edge_1", compute_capacity=10, network_latency=5, cost_per_request=0.01,),
    Node(name="edge_2", compute_capacity=20, network_latency=10, cost_per_request=0.015,),
    Node(name="cloud", compute_capacity=100, network_latency=50, cost_per_request=0.05,),
]


# Train XGBoost
print("Training XGBoost model...")
training_requests = generate_requests(1000)
dataset = generate_training_data(NODES, training_requests)
model, _, _ = train_model(dataset)

print("Model trained.\n")


# Run end-to-end experiment
results = []
requests = []
states = []
SEEDS = [42, 43, 44, 45, 46]

for request_id in range(NUM_REQUESTS):
    request = InferenceRequest(request_id=request_id, required_compute=2.0 + (request_id % 5) * 2.0,
                               deadline=50.0 + (request_id % 4) * 25.0,)
    for node in NODES:
        if "edge" in node.name:
            node.current_load = rng.uniform(0.0, 1.0) * node.compute_capacity
            node.network_latency = (node.base_network_latency* rng.uniform(0.5, 2.0))
        else:
            node.current_load = rng.uniform(0.0, 0.5) * node.compute_capacity
            node.network_latency = (node.base_network_latency* rng.uniform(0.5, 1.5))

    requests.append(request)

    states.append([(node.current_load, node.network_latency)
        for node in NODES
    ])

policies = {
    "Always Edge": lambda nodes, request: always_edge_baseline(nodes, request),
    "Always Cloud": lambda nodes, request: always_cloud_baseline(nodes, request),
    "Greedy": lambda nodes, request: greedy_baseline(nodes, request),
    "XGBoost": lambda nodes, request: predict_node(
        model, nodes, request,),
}

all_results = {}

for policy_name, policy in policies.items():

    results = []

    for request_id, request in enumerate(requests):
        # Restore exactly the same state for every policy
        for node, state in zip(NODES, states[request_id]):
            node.current_load = state[0]
            node.network_latency = state[1]
        scheduler_start = time.perf_counter()
        selected_node = policy(NODES,request,)

        scheduler_time_ms = (time.perf_counter() - scheduler_start) * 1000
        request_start = time.perf_counter()

        scheduler_start = time.perf_counter()
        selected_node = policy(NODES, request)
        scheduler_time_ms = (time.perf_counter() - scheduler_start) * 1000
        inference_result = send_inference(selected_node.name, IMAGE_PATH,)
        end_to_end_time_ms = (time.perf_counter() - request_start) * 1000
        deadline_met = (end_to_end_time_ms <= request.deadline)

        results.append({
            "node": selected_node.name,
            "scheduler_time_ms": scheduler_time_ms,
            "inference_time_ms": inference_result[
                "inference_time_ms"
            ],
            "api_time_ms": inference_result[
                "total_time_ms"
            ],
            "end_to_end_time_ms": end_to_end_time_ms,
            "deadline_met": deadline_met,
            "cost": selected_node.cost_per_request,
        })

    all_results[policy_name] = results

print("\n" + "=" * 70)
print("END-TO-END POLICY COMPARISON")
print("=" * 70)

for policy_name, results in all_results.items():

    latencies = [r["end_to_end_time_ms"] for r in results]
    inference_times = [r["inference_time_ms"] for r in results]
    scheduler_times = [r["scheduler_time_ms"] for r in results]
    violations = sum(not r["deadline_met"] for r in results)
    costs = [r["cost"] for r in results]

    print(f"\n{policy_name}")
    print(
        f"  End-to-end latency: "
        f"{statistics.mean(latencies):.2f} ± " f"{statistics.stdev(latencies):.2f} ms"
    )
    print(
        f"  Inference latency:  "
        f"{statistics.mean(inference_times):.2f} ± " f"{statistics.stdev(inference_times):.2f} ms"
    )
    print(
        f"  Scheduler time:     "
        f"{statistics.mean(scheduler_times):.2f} ± " f"{statistics.stdev(scheduler_times):.2f} ms"
    )
    print(
        f"  Deadline violations: "
        f"{violations}/{NUM_REQUESTS} " f"({violations / NUM_REQUESTS:.1%})"
    )

    print(
        f"  Average cost:       "
        f"{statistics.mean(costs):.3f}"
    )

    print("  Node selections:")

    for node in NODES:
        count = sum(r["node"] == node.name for r in results)
        print(f"    {node.name}: " f"{count}/{NUM_REQUESTS}")
    