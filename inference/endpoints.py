import os
from dotenv import load_dotenv

load_dotenv()

ENDPOINTS = {
    "edge_1": "http://localhost:8001/predict",
    "edge_2": "http://localhost:8002/predict",
    "cloud": os.getenv("CLOUD_ENDPOINT") + "/predict",
}