import cv2
from inference.tracker_engine import TrackerEngine

engine = TrackerEngine("models/best.pt")

def process_video_stream(video_path):
    cap = cv2.VideoCapture(video_path)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = engine.predict(frame)
        frame = engine.render(results)

        # encode frame for streaming
        _, buffer = cv2.imencode(".jpg", frame)
        frame = buffer.tobytes()

        yield frame

    cap.release()