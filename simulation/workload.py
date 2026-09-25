"""
Generate inference requests
"""
import random
from dataclasses import dataclass

@dataclass
class InferenceRequest:
    request_id: int
    required_compute: float
    deadline: float

def generate_requests(num_requests: int, seed: int=42) -> list[InferenceRequest]:
    """
    Generate rnd inference requests
    """
    rng = random.Random(seed)
    requests = []
    for i in range(num_requests):
        requests.append(
            InferenceRequest(
                request_id=i,
                required_compute=rng.uniform(2, 10), # a uniform distribution, any number in between
                deadline=rng.uniform(50, 200),
            )
        )
    return requests

if __name__ == "__main__":
    requests = generate_requests(10)

    for request in requests:
        print(request)