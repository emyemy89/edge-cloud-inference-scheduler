import torch
from PIL import Image
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

class MobileNet_Inference:
    def __init__(self):
        self.device = torch.device("cpu")