import time
import requests

from inference.endpoints import ENDPOINTS


def send_inference(node_name, image_path):
    url = ENDPOINTS[node_name]

    with open(image_path, "rb") as image_file:
        response = requests.post(
            url,
            files={"image": image_file},
        )

    if not response.ok:
        print("Status:", response.status_code)
        print("Response:", response.text)

    response.raise_for_status()
    return response.json()


def measure_network_latency(node_name, num_samples=3):
    url = ENDPOINTS[node_name].replace("/predict", "/docs")

    measurements = []

    for _ in range(num_samples):
        start = time.perf_counter()
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        elapsed = (time.perf_counter() - start) * 1000
        measurements.append(elapsed)

    return sum(measurements) / len(measurements)