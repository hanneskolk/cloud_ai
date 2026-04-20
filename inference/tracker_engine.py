from .tensorrt_engine import TensorRTEngine
from .onnx_engine import ONNXEngine
import cv2
import numpy as np

class TrackerEngine:

    def __init__(self, model_path, backend="onnx"):

        if backend == "tensorrt":
            raise NotImplementedError(
                "TensorRT backend is not yet implemented. Use backend='onnx'."
            )
        elif backend == "onnx":
            self.model = ONNXEngine(model_path)
        else:
            raise ValueError(f"Unknown backend '{backend}'. Choose 'onnx' or 'tensorrt'.")

    def process_frame(self, frame):
        return self.model.infer(frame)

    def process_detections(self, results, conf_threshold=0.25):
        # (1, 11, 8400) → (8400, 11)
        detections = results[0].T

        # Columns: [cx, cy, w, h, conf, cls_0 ... cls_N]
        boxes_xywh = detections[:, :4]
        confidences = detections[:, 4]
        class_scores = detections[:, 5:]

        # Filter by confidence
        mask = confidences > conf_threshold
        boxes_xywh = boxes_xywh[mask]
        confidences = confidences[mask]
        class_scores = class_scores[mask]

        if len(boxes_xywh) == 0:
            return []

        # Get class with highest score per detection
        class_ids = np.argmax(class_scores, axis=1)

        # Convert cx, cy, w, h → x1, y1, x2, y2
        x1 = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2
        y1 = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2
        x2 = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2
        y2 = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2

        return list(zip(x1, y1, x2, y2, confidences, class_ids))

    def render(self, frame, results):
        if results is None or len(results) == 0:
            return frame

        orig_h, orig_w = frame.shape[:2]

        # Scale factors from 640×640 model input back to original frame size
        scale_x = orig_w / 640.0
        scale_y = orig_h / 640.0

        detections = self.process_detections(results)

        for (x1, y1, x2, y2, conf, cls_id) in detections:
            # Scale boxes back to original frame dimensions
            x1 = int(x1 * scale_x)
            y1 = int(y1 * scale_y)
            x2 = int(x2 * scale_x)
            y2 = int(y2 * scale_y)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"cls{cls_id} {conf:.2f}",
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 255, 0), 1
            )

        return frame