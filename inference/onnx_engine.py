import onnxruntime as ort
import numpy as np
import cv2

from .base_engine import BaseEngine

DTYPE_MAP = {
    "tensor(float)": np.float32,
    "tensor(float16)": np.float16,
    "tensor(uint8)": np.uint8,
}

class ONNXEngine(BaseEngine):

    def __init__(self, model_path):
        self.session = ort.InferenceSession(
            model_path,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        
        input_meta = self.session.get_inputs()[0]
        shape = input_meta.shape
        self.input_h = shape[2] if isinstance(shape[2], int) else 640
        self.input_w = shape[3] if isinstance(shape[3], int) else 640
        
        self.input_dtype = DTYPE_MAP.get(input_meta.type, np.float32)

    def preprocess(self, frame):
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.input_w, self.input_h))
        img = img.astype(self.input_dtype) / 255.0
        img = np.transpose(img, (2, 0, 1))
        return np.expand_dims(img, axis=0)

    def infer(self, frame):
        input_tensor = self.preprocess(frame)

        outputs = self.session.run(
            None,
            {self.session.get_inputs()[0].name: input_tensor}
        )

        return outputs[0]