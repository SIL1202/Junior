# YOLOv11-pose inference example

import torch
from ultralytics import YOLO

model = YOLO("yolo11n-pose.pt")
# features = []


# def hook_fn(model, input, output):
#     features.append(output.detach().cpu())


# layer = model.model.[10]

predict = model.predict(source="../DataSet/source1.mp4", save=True, show=True)
