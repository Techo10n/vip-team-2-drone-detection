# train.py
from ultralytics import YOLO

# Load YOLOv8 nano model (fast for small datasets)
model = YOLO("yolov8n.pt")

# Train the model
model.train(
    data="data.yaml",        # your YAML with train/val paths
    epochs=15,               # shorter for faster demo
    imgsz=480,               # small images → faster training
    batch=16,                # safe batch size; increase if GPU memory allows
    name="human_detector",   # folder name for this run
    project="runs/train",    # default YOLO project folder
    workers=4,               # CPU threads for data loading
    exist_ok=True            # overwrite previous results if folder exists
)

print("Training started! Check runs/train/human_detector for results, validation predictions, and metrics.")