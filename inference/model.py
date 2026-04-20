from ultralytics import YOLO

MODEL_PATH = "models/best.pt"

model = YOLO(MODEL_PATH)

def run_inference(frame):
    results = model(frame, conf=0.2)[0]
    return results