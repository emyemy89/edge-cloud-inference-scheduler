"""
Generate inference requests
"""
from dataclasses import dataclass

@dataclass
class InferenceRequest:
    request_id: int
    required_compute: float
    deadline: float

    def generate_requests(num_requests: int) -> list[InferenceRequest]:
        """
        Generate requests based on num_requests
        """
        return [
            InferenceRequest(
                request_id=i,
                required_compute=5.0,
                deadline=100.0,
            )
            for i in range(num_requests)
        ]