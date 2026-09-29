import time
import torch
from PIL import Image
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

class MobileNetInference:
    def __init__(self):
        self.device = torch.device("cpu")
        weights = MobileNet_V3_Small_Weights.DEFAULT

        self.model = mobilenet_v3_small(weights=weights)
        self.model.eval()
        self.model.to(self.device)

        self.preprocess = weights.transforms()
        self.categories = weights.meta["categories"]

    def predict(self, image_path: str):
        image = Image.open(image_path).convert("RGB")
        input_tensor = self.preprocess(image).unsqueeze(0)
        input_tensor = input_tensor.to(self.device)
        start_time = time.perf_counter()
        with torch.no_grad(): # No gradient needed for inference
            output = self.model(input_tensor)
        inference_time = (time.perf_counter() - start_time) * 1000
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        confidence, class_index = torch.max(probabilities, dim=0)
        return {
            "class": self.categories[class_index.item()],
            "confidence": confidence.item(),
            "inference_time_ms": inference_time,
        }

if __name__ == "__main__":
    model = MobileNetInference()
    result = model.predict("test.jpg")
    print(f"Class: {result['class']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"Inference time: " f"{result['inference_time_ms']:.2f} ms")
