import cv2
from ultralytics import YOLO

class TrackerEngine:
    def __init__(self, model_path: str, conf: float = 0.25, imgsz: int = 640):
        self.model = YOLO(model_path)
        self.conf = conf
        self.imgsz = imgsz

    def process_frame(self, frame):
        """
        Runs YOLO + ByteTrack in one call.
        Returns annotated frame and raw results.
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

    def draw(self, results):
        """
        Uses Ultralytics built-in visualization (fastest path)
        """
        return results.plot()