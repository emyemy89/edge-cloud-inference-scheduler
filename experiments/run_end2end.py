import time
import statistics
import random
from pathlib import Path

from inference.client import send_inference, measure_network_latency
from ml.dataset import generate_training_data
from ml.xgboost_scheduler import train_model, predict_node
from nodes.node import create_nodes
from simulation.workload import InferenceRequest, generate_requests
from scheduler.baselines import always_edge_baseline, always_cloud_baseline, greedy_baseline


# --------------------------------------------------
# Configuration

IMAGE_PATH = Path("../inference/images/dog.png")
NUM_REQUESTS = 50

# Same basic node configuration as the simulator
NODES = create_nodes()


# Train XGBoost
print("Training XGBoost model...")
training_requests = generate_requests(1000)
dataset = generate_training_data(NODES, training_requests)
model, _, _ = train_model(dataset)

print("Model trained.\n")

policies = {
    "Always Edge": lambda nodes, request: always_edge_baseline(nodes, request),
    "Always Cloud": lambda nodes, request: always_cloud_baseline(nodes, request),
    "Greedy": lambda nodes, request: greedy_baseline(nodes, request),
    "XGBoost": lambda nodes, request: predict_node(model, nodes, request,),
}


# Run end-to-end experiment
SEEDS = [42, 43, 44, 45, 46]
all_seed_results = {}

for seed in SEEDS:
    rng = random.Random(seed)
    requests = []
    states = []
    for request_id in range(NUM_REQUESTS):
        request = InferenceRequest(
            request_id=request_id,
            required_compute=2.0 + (request_id % 5) * 2.0,
            deadline=50.0 + (request_id % 4) * 25.0,
        )

        for node in NODES:
            if "edge" in node.name:
                node.current_load = rng.uniform(0.0, 1.0) * node.compute_capacity
            else:
                node.current_load = rng.uniform(0.0, 0.5) * node.compute_capacity

        requests.append(request)
        states.append([(node.current_load, node.network_latency) for node in NODES])

    all_results = {}


    for policy_name, policy in policies.items():

        results = []
        for request_id, request in enumerate(requests):
            # Restore exactly the same state for every policy
            for node, state in zip(NODES, states[request_id]):
                node.current_load = state[0]
                node.network_latency = state[1]

            request_start = time.perf_counter()
            for node in NODES:
                node.network_latency = measure_network_latency(node.name, num_samples=1)
            scheduler_start = time.perf_counter()
            selected_node = policy(NODES, request)
            scheduler_time_ms = (time.perf_counter() - scheduler_start) * 1000
            inference_result = send_inference(selected_node.name, IMAGE_PATH,)
            end_to_end_time_ms = (time.perf_counter() - request_start) * 1000
            deadline_met = (end_to_end_time_ms <= request.deadline)

            results.append({
                "node": selected_node.name,
                "scheduler_time_ms": scheduler_time_ms,
                "inference_time_ms": inference_result["inference_time_ms"],
                "api_time_ms": inference_result["total_time_ms"],
                "end_to_end_time_ms": end_to_end_time_ms,
                "deadline_met": deadline_met,
                "cost": selected_node.cost_per_request,
            })

        all_results[policy_name] = results
    all_seed_results[seed] = all_results

print("\n" + "=" * 70)
print("END-TO-END POLICY COMPARISON")
print("=" * 70)

for policy_name in policies:

    seed_latencies = []
    seed_inference_times = []
    seed_scheduler_times = []
    seed_costs = []
    seed_violation_rates = []

    for seed in SEEDS:
        results = all_seed_results[seed][policy_name]

        latencies = [r["end_to_end_time_ms"] for r in results]
        inference_times = [r["inference_time_ms"] for r in results]
        scheduler_times = [r["scheduler_time_ms"] for r in results]
        costs = [r["cost"] for r in results]

        violations = sum(not r["deadline_met"] for r in results)

        seed_latencies.append(statistics.mean(latencies))
        seed_inference_times.append(statistics.mean(inference_times))
        seed_scheduler_times.append(statistics.mean(scheduler_times))
        seed_costs.append(statistics.mean(costs))
        seed_violation_rates.append(violations / NUM_REQUESTS)

    print(f"\n{policy_name}")

    print(
        f"  End-to-end latency: "
        f"{statistics.mean(seed_latencies):.2f} ± " f"{statistics.stdev(seed_latencies):.2f} ms"
    )

    print(
        f"  Inference latency:  "
        f"{statistics.mean(seed_inference_times):.2f} ± " f"{statistics.stdev(seed_inference_times):.2f} ms"
    )

    print(
        f"  Scheduler time:     "
        f"{statistics.mean(seed_scheduler_times):.2f} ± " f"{statistics.stdev(seed_scheduler_times):.2f} ms"
    )

    print(
        f"  Deadline violations: "
        f"{statistics.mean(seed_violation_rates):.1%} ± " f"{statistics.stdev(seed_violation_rates):.1%}"
    )

    print(
        f"  Average cost:       "
        f"{statistics.mean(seed_costs):.3f} ± " f"{statistics.stdev(seed_costs):.3f}"
    )
    print("  Node selections:")

    for node in NODES:
        count = sum(r["node"] == node.name for r in results)
        print(f"    {node.name}: " f"{count}/{NUM_REQUESTS}")
    