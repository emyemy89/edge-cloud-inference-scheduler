import requests

from inference.endpoints import ENDPOINTS


def send_inference(node_name, image_path):
    url = ENDPOINTS[node_name]
    with open(image_path, "rb") as image_file:
        response = requests.post(url, files={"file": image_file})
    response.raise_for_status()
    return response.json()