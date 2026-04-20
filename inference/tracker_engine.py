from .tensorrt_engine import TensorRTEngine
from .onnx_engine import ONNXEngine
import cv2

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

    def render(self, frame, results):
        print("results shape:", results.shape)
        print("first row:", results[0])
        if results is None or len(results) == 0:
            return frame

        for det in results:
            if len(det) < 6:
                continue
            x1, y1, x2, y2, conf, cls = det[:6]
            cv2.rectangle(
                frame,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                (0, 255, 0), 2
            )
            cv2.putText(
                frame,
                f"{conf:.2f}",
                (int(x1), int(y1) - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 255, 0), 1
            )

        return frame