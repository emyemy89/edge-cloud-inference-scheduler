"""
XGBoost hyperparameter tuning experiment.

Tests each hyperparameter independently, then combines
the best-performing values and evaluates the final model.
"""

import random
import statistics

from nodes.node import Node
from simulation.workload import generate_requests
from simulation.environment import SimulationEnvironment
from simulation.network import update_network_conditions, NETWORK_UPDATE_INTERVAL
from evaluation.metrics import Metrics

from ml.dataset import generate_training_data
from ml.xgboost_scheduler import train_model, predict_node


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

TRAINING_REQUESTS = 1000
EVALUATION_REQUESTS = 100

EVALUATION_SEEDS = [1, 2, 3, 4, 5]

WORKLOAD_SCENARIOS = {
    "low_load": 50,
    "medium_load": 20,
    "high_load": 10,
    "very_high_load": 5,
}

NETWORK_SCENARIOS = [
    "stable",
    "moderate",
    "high",
]


# ---------------------------------------------------------
# Nodes
# ---------------------------------------------------------

def create_nodes() -> list[Node]:
    return [
        Node(
            name="edge_1",
            compute_capacity=10,
            network_latency=5,
            cost_per_request=0.01,
        ),
        Node(
            name="edge_2",
            compute_capacity=20,
            network_latency=10,
            cost_per_request=0.015,
        ),
        Node(
            name="cloud",
            compute_capacity=100,
            network_latency=50,
            cost_per_request=0.05,
        ),
    ]


# ---------------------------------------------------------
# Training
# ---------------------------------------------------------

def train_tuned_model(dataset, params):
    """
    Train XGBoost with the supplied hyperparameters.
    """

    from xgboost import XGBClassifier

    X = dataset.drop(columns="target")
    y = dataset["target"].map({
        "edge_1": 0,
        "edge_2": 1,
        "cloud": 2,
    })

    model = XGBClassifier(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        learning_rate=params["learning_rate"],
        min_child_weight=params["min_child_weight"],
        subsample=params["subsample"],
        colsample_bytree=params["colsample_bytree"],
        gamma=params["gamma"],
        reg_lambda=params["reg_lambda"],
        random_state=42,
        eval_metric="mlogloss",
    )

    model.fit(X, y)

    return model


# ---------------------------------------------------------
# Simulation
# ---------------------------------------------------------

def run_ml_policy(model, requests, arrival_interval, network_scenario, seed=42):
    nodes = create_nodes()
    environment = SimulationEnvironment(nodes)
    metrics = Metrics()

    rng = random.Random(seed)

    for i, request in enumerate(requests):

        if i % NETWORK_UPDATE_INTERVAL == 0:
            update_network_conditions(
                nodes,
                network_scenario,
                rng,
            )

        environment.release_finished_requests()

        features = predict_node(model, nodes, request)
        selected_node = features

        latency = environment.execute(
            selected_node,
            request,
        )

        metrics.record(
            request_id=request.request_id,
            node_name=selected_node.name,
            latency=latency,
            cost=selected_node.cost_per_request,
            utilization=selected_node.utilization(),
            deadline=request.deadline,
        )

        environment.advance_time(arrival_interval)

    return metrics


# ---------------------------------------------------------
# Evaluation
# ---------------------------------------------------------

def evaluate_model_on_simulation(model, dataset_name):
    """
    Run the model over all 12 scenarios and 5 seeds.
    Returns overall averages used for comparing configurations.
    """

    results = []

    for network_scenario in NETWORK_SCENARIOS:

        for scenario_name, arrival_interval in WORKLOAD_SCENARIOS.items():

            for seed in EVALUATION_SEEDS:

                requests = generate_requests(
                    EVALUATION_REQUESTS,
                    seed=seed,
                )

                metrics = run_ml_policy(
                    model,
                    requests,
                    arrival_interval,
                    network_scenario,
                    seed=seed,
                )

                results.append({
                    "latency": metrics.average_latency(),
                    "cost": metrics.total_cost(),
                    "utilization": metrics.average_utilization(),
                    "violations": metrics.deadline_violation_rate(),
                })

    mean_latency = statistics.mean(
        result["latency"] for result in results
    )

    mean_cost = statistics.mean(
        result["cost"] for result in results
    )

    mean_utilization = statistics.mean(
        result["utilization"] for result in results
    )

    mean_violations = statistics.mean(
        result["violations"] for result in results
    )

    print(
        f"{dataset_name:<25} "
        f"latency={mean_latency:7.2f} ms | "
        f"cost={mean_cost:6.2f} | "
        f"util={mean_utilization:6.2%} | "
        f"violations={mean_violations:6.2%}"
    )

    return {
        "latency": mean_latency,
        "cost": mean_cost,
        "utilization": mean_utilization,
        "violations": mean_violations,
    }


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    print("\n==============================")
    print("XGBOOST HYPERPARAMETER TUNING")
    print("==============================")

    print("\nGenerating training data...")

    training_nodes = create_nodes()
    training_requests = generate_requests(
        TRAINING_REQUESTS,
        seed=42,
    )

    dataset = generate_training_data(
        training_nodes,
        training_requests,
    )

    # -----------------------------------------------------
    # Baseline configuration
    # -----------------------------------------------------

    baseline_params = {
        "n_estimators": 100,
        "max_depth": 4,
        "learning_rate": 0.1,
        "min_child_weight": 1,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
        "gamma": 0,
        "reg_lambda": 1,
    }

    print("\n---- BASELINE ----")

    best_params = baseline_params.copy()

    best_result = evaluate_model_on_simulation(
        train_tuned_model(dataset, best_params),
        "Baseline",
    )

    # -----------------------------------------------------
    # 1. n_estimators
    # -----------------------------------------------------

    print("\n---- TESTING n_estimators ----")

    for value in [50, 100, 200, 300, 500]:

        params = best_params.copy()
        params["n_estimators"] = value

        result = evaluate_model_on_simulation(
            train_tuned_model(dataset, params),
            f"n_estimators={value}",
        )

        if result["latency"] < best_result["latency"]:
            best_params = params
            best_result = result

    print(f"Selected n_estimators: {best_params['n_estimators']}")

    # -----------------------------------------------------
    # 2. learning_rate
    # -----------------------------------------------------

    print("\n---- TESTING learning_rate ----")

    for value in [0.01, 0.03, 0.05, 0.1, 0.15]:

        params = best_params.copy()
        params["learning_rate"] = value

        result = evaluate_model_on_simulation(
            train_tuned_model(dataset, params),
            f"learning_rate={value}",
        )

        if result["latency"] < best_result["latency"]:
            best_params = params
            best_result = result

    print(f"Selected learning_rate: {best_params['learning_rate']}")

    # -----------------------------------------------------
    # 3. max_depth
    # -----------------------------------------------------

    print("\n---- TESTING max_depth ----")

    for value in [2, 3, 4, 5, 6]:

        params = best_params.copy()
        params["max_depth"] = value

        result = evaluate_model_on_simulation(
            train_tuned_model(dataset, params),
            f"max_depth={value}",
        )

        if result["latency"] < best_result["latency"]:
            best_params = params
            best_result = result

    print(f"Selected max_depth: {best_params['max_depth']}")

    # -----------------------------------------------------
    # 4. min_child_weight
    # -----------------------------------------------------

    print("\n---- TESTING min_child_weight ----")

    for value in [1, 2, 3, 5]:

        params = best_params.copy()
        params["min_child_weight"] = value

        result = evaluate_model_on_simulation(
            train_tuned_model(dataset, params),
            f"min_child_weight={value}",
        )

        if result["latency"] < best_result["latency"]:
            best_params = params
            best_result = result

    print(
        f"Selected min_child_weight: "
        f"{best_params['min_child_weight']}"
    )

    # -----------------------------------------------------
    # 5. subsample
    # -----------------------------------------------------

    print("\n---- TESTING subsample ----")

    for value in [0.7, 0.8, 0.9, 1.0]:

        params = best_params.copy()
        params["subsample"] = value

        result = evaluate_model_on_simulation(
            train_tuned_model(dataset, params),
            f"subsample={value}",
        )

        if result["latency"] < best_result["latency"]:
            best_params = params
            best_result = result

    print(f"Selected subsample: {best_params['subsample']}")

    # -----------------------------------------------------
    # 6. colsample_bytree
    # -----------------------------------------------------

    print("\n---- TESTING colsample_bytree ----")

    for value in [0.7, 0.8, 0.9, 1.0]:

        params = best_params.copy()
        params["colsample_bytree"] = value

        result = evaluate_model_on_simulation(
            train_tuned_model(dataset, params),
            f"colsample_bytree={value}",
        )

        if result["latency"] < best_result["latency"]:
            best_params = params
            best_result = result

    print(
        f"Selected colsample_bytree: "
        f"{best_params['colsample_bytree']}"
    )

    # -----------------------------------------------------
    # 7. gamma
    # -----------------------------------------------------

    print("\n---- TESTING gamma ----")

    for value in [0, 0.1, 0.3, 0.5]:

        params = best_params.copy()
        params["gamma"] = value

        result = evaluate_model_on_simulation(
            train_tuned_model(dataset, params),
            f"gamma={value}",
        )

        if result["latency"] < best_result["latency"]:
            best_params = params
            best_result = result

    print(f"Selected gamma: {best_params['gamma']}")

    # -----------------------------------------------------
    # 8. reg_lambda
    # -----------------------------------------------------

    print("\n---- TESTING reg_lambda ----")

    for value in [1, 2, 5, 10]:

        params = best_params.copy()
        params["reg_lambda"] = value

        result = evaluate_model_on_simulation(
            train_tuned_model(dataset, params),
            f"reg_lambda={value}",
        )

        if result["latency"] < best_result["latency"]:
            best_params = params
            best_result = result

    print(f"Selected reg_lambda: {best_params['reg_lambda']}")

    # -----------------------------------------------------
    # Final combined model
    # -----------------------------------------------------

    print("\n==============================")
    print("FINAL COMBINED PARAMETERS")
    print("==============================")

    for parameter, value in best_params.items():
        print(f"{parameter}: {value}")

    print("\n---- FINAL MODEL ----")

    final_model = train_tuned_model(
        dataset,
        best_params,
    )

    evaluate_model_on_simulation(
        final_model,
        "Combined tuned model",
    )


if __name__ == "__main__":
    main()