from ultralytics import YOLO

model = YOLO("yolo11n-pose.pt")  # load a pretrained model (recommended for training)

results = model.predict(source="source.mp4", show=True, save=True)
