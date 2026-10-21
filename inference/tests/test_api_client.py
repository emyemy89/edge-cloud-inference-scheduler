from inference.client import send_inference


image_path = "../images/dog.png"
for node in ["edge_1", "edge_2", "cloud"]:
    result = send_inference(node, image_path)
    print(f"\n{node}")
    print(result)