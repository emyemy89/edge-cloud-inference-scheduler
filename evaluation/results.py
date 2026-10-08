import statistics

from evaluation.metrics import Metrics


def aggregate_results(results):
    for key, runs in results.items():
        latencies = [run["latency"] for run in runs]
        costs = [run["cost"] for run in runs]
        utilizations = [run["utilization"] for run in runs]
        violations = [run["violations"] for run in runs]

        print(f"\n{' / '.join(key)}")
        print(
            f"Average latency: "
            f"{statistics.mean(latencies):.2f} ± "
            f"{statistics.stdev(latencies):.2f} ms"
        )
        print(
            f"Average cost: "
            f"{statistics.mean(costs):.2f} ± "
            f"{statistics.stdev(costs):.2f}"
        )
        print(
            f"Average utilization: "
            f"{statistics.mean(utilizations):.2%} ± "
            f"{statistics.stdev(utilizations):.2%}"
        )
        print(
            f"Average deadline violations: "
            f"{statistics.mean(violations):.2%} ± "
            f"{statistics.stdev(violations):.2%}"
        )

def print_results(policy_name: str, metrics: Metrics, seed):
    print(
        f"Seed {seed} - {policy_name}: "
        f"latency={metrics.average_latency():.2f} ms, "
        f"cost={metrics.total_cost():.2f}, "
        f"utilization={metrics.average_utilization():.2%}, "
        f"violations={metrics.deadline_violation_rate():.2%}"
    )
