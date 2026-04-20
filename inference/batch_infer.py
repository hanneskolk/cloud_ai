import cv2
from inference.model import run_inference
from tracker.bytetrack_wrapper import track
from utils.draw import draw_boxes

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
        detections = results.boxes.xyxy.cpu().numpy()

        tracks = track(detections, frame)

        frame = draw_boxes(frame, tracks)
        out.write(frame)

    cap.release()
    out.release()

    return output_path