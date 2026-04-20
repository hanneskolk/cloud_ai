import cv2
from ultralytics import YOLO

class TrackerEngine:
    def __init__(self, model_path, conf=0.25, imgsz=640):
        self.model = YOLO(model_path)
        self.conf = conf
        self.imgsz = imgsz

    def process_frame(self, frame):
        """
        Runs YOLO + ByteTrack in one call.
        """
        results = self.model.track(
            frame,
            persist=True,
            conf=self.conf,
            imgsz=self.imgsz,
            tracker="bytetrack.yaml",
            verbose=False
        )[0]

        return results

    def render(self, results):
        """
        Fast built-in visualization
        """
        return results.plot()