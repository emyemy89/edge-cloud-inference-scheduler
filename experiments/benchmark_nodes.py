import statistics
import time
from pathlib import Path

from inference.client import send_inference


IMAGE_PATH = Path("../inference/dog.png")
NUM_REQUESTS = 20

NODES = ["edge_1", "edge_2", "cloud"]


for node in NODES:
    inference_times = []
    api_times = []
    end_to_end_times = []
    print(f"\nBenchmarking {node}...")

    for i in range(NUM_REQUESTS):
        start = time.perf_counter()
        result = send_inference(node, IMAGE_PATH,)

        end_to_end = (time.perf_counter() - start) * 1000

        inference_times.append(result["inference_time_ms"])
        api_times.append(result["total_time_ms"])

        end_to_end_times.append(end_to_end)

    print(f"  Inference:  " f"{statistics.mean(inference_times):.2f} ± " f"{statistics.stdev(inference_times):.2f} ms")

    print(f"  API:        "f"{statistics.mean(api_times):.2f} ± "f"{statistics.stdev(api_times):.2f} ms")

    print(f"  End-to-end: " f"{statistics.mean(end_to_end_times):.2f} ± "f"{statistics.stdev(end_to_end_times):.2f} ms")