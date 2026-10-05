import time
import random
from pathlib import Path

from nodes.node import Node
from simulation.workload import InferenceRequest, generate_requests
from ml.dataset import generate_training_data
from ml.xgboost_scheduler import train_model, predict_node
from inference.client import send_inference


# --------------------------------------------------
# Configuration

IMAGE_PATH = Path("../inference/dog.png")
NUM_REQUESTS = 20

# Same basic node configuration as the simulator
NODES = [
    Node(name="edge_1", compute_capacity=10, network_latency=5, cost_per_request=0.01,),
    Node(name="edge_2", compute_capacity=20, network_latency=10, cost_per_request=0.015,),
    Node(name="cloud", compute_capacity=100, network_latency=50, cost_per_request=0.05,),
]


# Train XGBoost
print("Training XGBoost model...")
requests = generate_requests(1000)
dataset = generate_training_data(NODES, requests)
model, _, _ = train_model(dataset)

print("Model trained.\n")


# Run end-to-end experiment
results = []
rng = random.Random(42)

for request_id in range(NUM_REQUESTS):
    request = InferenceRequest(request_id=request_id, required_compute=2.0 + (request_id % 5) * 2.0,
                               deadline=50.0 + (request_id % 4) * 25.0,)

    # Measure only the scheduler decision
    scheduler_start = time.perf_counter()

    for node in NODES:
        if "edge" in node.name:
            node.current_load = rng.uniform(0.0, 1.0) * node.compute_capacity
            node.network_latency = (node.base_network_latency* rng.uniform(0.5, 2.0))
        else:
            node.current_load = rng.uniform(0.0, 0.5) * node.compute_capacity
            node.network_latency = (node.base_network_latency* rng.uniform(0.5, 1.5))

    selected_node = predict_node( model, NODES, request,)
    scheduler_time_ms = (time.perf_counter() - scheduler_start) * 1000

    # Send actual inference request
    inference_start = time.perf_counter()

    inference_result = send_inference(selected_node.name, IMAGE_PATH,)

    end_to_end_time_ms = (time.perf_counter() - inference_start) * 1000

    result = {
        "request_id": request_id,
        "node": selected_node.name,
        "scheduler_time_ms": scheduler_time_ms,
        "inference_time_ms": inference_result["inference_time_ms"],
        "api_total_time_ms": inference_result["total_time_ms"],
        "end_to_end_time_ms": end_to_end_time_ms,
        "class": inference_result["class"],
        "confidence": inference_result["confidence"],
    }

    results.append(result)

    print(
        f"Request {request_id + 1:02d}: "
        f"{selected_node.name:7s} | "
        f"compute={request.required_compute:.1f} | "
        f"deadline={request.deadline:.1f} ms | "
        f"scheduler={scheduler_time_ms:.2f} ms | "
        f"inference={inference_result['inference_time_ms']:.2f} ms | "
        f"end-to-end={end_to_end_time_ms:.2f} ms"
    )


# --------------------------------------------------
# Aggregate results

print("\n" + "=" * 60)
print("END-TO-END RESULTS")
print("=" * 60)

avg_scheduler = sum(r["scheduler_time_ms"] for r in results) / len(results)

avg_inference = sum(r["inference_time_ms"] for r in results) / len(results)

avg_api = sum(r["api_total_time_ms"] for r in results) / len(results)

avg_end_to_end = sum(r["end_to_end_time_ms"] for r in results) / len(results)

print(f"Requests:                {NUM_REQUESTS}")
print(f"Average scheduler time:  {avg_scheduler:.2f} ms")
print(f"Average inference time:  {avg_inference:.2f} ms")
print(f"Average API time:        {avg_api:.2f} ms")
print(f"Average end-to-end time: {avg_end_to_end:.2f} ms")

print("\nNode selections:")

for node in NODES:
    count = sum(r["node"] == node.name for r in results)

    print(f"  {node.name}: {count}/{NUM_REQUESTS}")
    