from .tensorrt_engine import TensorRTEngine
from .onnx_engine import ONNXEngine

class TrackerEngine:

    def __init__(self, model_path, backend="tensorrt"):

        if backend == "tensorrt":
            self.model = TensorRTEngine(model_path)
        else:
            self.model = ONNXEngine(model_path)

    def process_frame(self, frame):
        return self.model.infer(frame)

    def render(self, frame, results):
        # keep your existing visualization logic
        # (boxes, labels, etc.)
        return frame