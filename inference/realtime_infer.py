import cv2
from inference.model import run_inference
from tracker.bytetrack_wrapper import track
from utils.draw import draw_boxes

def run_stream(stream_url):
    cap = cv2.VideoCapture(stream_url)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 640))

        results = run_inference(frame)
        detections = results.boxes.xyxy.cpu().numpy()

        tracks = track(detections, frame)
        frame = draw_boxes(frame, tracks)

        cv2.imshow("Live", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()