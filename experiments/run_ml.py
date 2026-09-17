"""
The ML-Scheduler simulation
"""
import pandas as pd

from nodes.node import Node
from simulation.workload import generate_requests
from ml.dataset import get_state_features, generate_training_data
from ml.xgboost_scheduler import predict_node, train_model

def create_nodes() -> list[Node]:
    return [
        Node(name="edge_1",
             compute_capacity=10,
             network_latency=5,
             cost_per_request=0.01),
        Node(name="edge_2",
             compute_capacity=20,
             network_latency=10,
             cost_per_request=0.015, ),
        Node(name="cloud",
             compute_capacity=100,
             network_latency=50,
             cost_per_request=0.05 ),
    ]

nodes = create_nodes()
requests = generate_requests(1000)
dataset = generate_training_data(nodes, requests)
model, X_test, y_test = train_model(dataset)

def xgboost_policy(nodes, request):
    features = get_state_features(nodes, request)
    X = pd.DataFrame([features])
    node_name = predict_node(model, X)
    return next(node for node in nodes if node.name == node_name)