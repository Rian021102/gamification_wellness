import ultralytics
from ultralytics import YOLO

model = YOLO("yolo11n-pose.pt")
device='mps'
model.to(device)
results=model.train(
    data='/Users/rianrachmanto/miniforge3/project/gamification_wellness/pushuppose.v1i.yolov8/data.yaml',
    imgsz=640,
    epochs=5,
    batch=16,
    project='yolov11n_pose',
    device=device
)
