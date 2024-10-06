import ultralytics
from ultralytics import YOLO

model = YOLO("yolo11n.pt")
device='mps'
model.to(device)

results=model.train(
    data='/Users/rianrachmanto/miniforge3/project/gamification_wellness/push up-ditection.v4i.yolov11/data.yaml',
    imgsz=320,
    epochs=5,
    batch=16,
    project='yolov11n_custom',
    device=device
)
