import cv2
from inference.model import run_inference
from tracker.bytetrack_wrapper import track
from utils.draw import draw_boxes
import numpy as np

def convert_detections(results):
    boxes = results.boxes.xyxy.cpu().numpy()
    scores = results.boxes.conf.cpu().numpy()

    detections = []
    for box, score in zip(boxes, scores):
        x1, y1, x2, y2 = box
        detections.append([float(x1), float(y1), float(x2), float(y2), float(score)])
        
    return detections

def process_video(input_path, output_path):
    cap = cv2.VideoCapture(input_path)

    width = int(cap.get(3))
    height = int(cap.get(4))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    out = cv2.VideoWriter(output_path,
                          cv2.VideoWriter_fourcc(*'mp4v'),
                          fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = run_inference(frame)
        detections = convert_detections(results)

        tracks = track(detections, frame)

        frame = draw_boxes(frame, tracks)
        out.write(frame)

    cap.release()
    out.release()

    return output_path