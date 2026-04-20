from ultralytics import YOLO

class TrackerEngine:
    def __init__(self, model_path, conf=0.15, imgsz=640):
        self.model = YOLO(model_path)
        self.conf = conf
        self.imgsz = imgsz

    def process(self, frame):
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
        return results.plot()