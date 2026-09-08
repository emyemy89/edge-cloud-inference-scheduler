"""
Generate inference requests
"""
from dataclasses import dataclass

@dataclass
class InferenceRequest:
    request_id: int
    required_compute: float
    deadline: float
    