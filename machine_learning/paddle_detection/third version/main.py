import torch
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

features = []

def hook_fn(model, input, output):
    features.append(output.detach().cpu())

layer = model.model.[10]

