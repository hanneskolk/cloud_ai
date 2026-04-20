import cv2
from inference.tracker_engine import TrackerEngine

def frame_generator(video_source, model_path):
    engine = TrackerEngine(model_path)

    cap = cv2.VideoCapture(video_source)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # optional speed boost
        frame = cv2.resize(frame, (640, 640))

        results = engine.process(frame)
        frame = engine.render(results)

        # encode frame for Streamlit
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield frame_bytes

    cap.release()