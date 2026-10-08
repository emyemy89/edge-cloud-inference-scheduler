from pathlib import Path

from nodes.node import Node
from simulation.workload import InferenceRequest, generate_requests
from ml.dataset import generate_training_data
from ml.xgboost_scheduler import train_model, predict_node
from inference.client import send_inference


# Configuration
IMAGE_PATH = Path("../images/dog.png")


# Create nodes
nodes = [
    Node(name="edge_1", compute_capacity=10, network_latency=5, cost_per_request=0.01,),
    Node(name="edge_2", compute_capacity=20, network_latency=10, cost_per_request=0.015,),
    Node(name="cloud", compute_capacity=100, network_latency=50, cost_per_request=0.05,),
]

# Train XGBoost model
requests = generate_requests(1000, seed=42)
dataset = generate_training_data(nodes, requests)

model, _, _ = train_model(dataset)

# Create one inference request
request = InferenceRequest(request_id=0, required_compute=5.0, deadline=100.0,)


# XGBoost selects the node
selected_node = predict_node(model, nodes, request,)
print(f"XGBoost selected: {selected_node.name}")


# Send image to selected Docker container
result = send_inference( selected_node.name, IMAGE_PATH,)
print("\nInference result:")
print(result)
