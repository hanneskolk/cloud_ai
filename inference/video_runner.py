import cv2
from inference.tracker_engine import TrackerEngine
import os

def process_video(input_path, output_path, model_path):
    
    output_path = os.path.abspath(output_path)

    engine = TrackerEngine(model_path)

    cap = cv2.VideoCapture(input_path)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or fps is None:
        fps = 25

    out = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    frame_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ⚡ performance optimization (optional)
        if frame_id % 2 != 0:
            frame_id += 1
            continue

        results = engine.process_frame(frame)
        frame = engine.render(results)

        out.write(frame)
        frame_id += 1

    cap.release()
    out.release()

    return output_path