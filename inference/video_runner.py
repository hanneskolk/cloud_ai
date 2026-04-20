import cv2
from inference.tracker_engine import TrackerEngine
import os
import subprocess

def process_video(input_path, output_path, model_path):
    
    output_path = os.path.abspath(output_path)

    engine = TrackerEngine(model_path, backend="onnx")

    cap = cv2.VideoCapture(input_path)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or fps is None:
        fps = 25
    
    temp_path = output_path.replace(".mp4", "_temp.avi")

    out = cv2.VideoWriter(
        temp_path,
        cv2.VideoWriter_fourcc(*"XVID"),
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
        frame = engine.render(frame, results)

        out.write(frame)
        print(frame_id)
        frame_id += 1
        
        

    cap.release()
    out.release()
    
    subprocess.run([
        "ffmpeg",
        "-y",
        "-i", temp_path,
        "-vcodec", "libx264",
        "-pix_fmt", "yuv420p",
        output_path
    ])
    
    print("File exists:", os.path.exists(output_path))
    print("File size:", os.path.getsize(output_path))

    return output_path