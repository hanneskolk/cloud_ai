import onnxruntime as ort
import numpy as np
import cv2

from .base_engine import BaseEngine

class ONNXEngine(BaseEngine):

    def __init__(self, model_path):
        self.session = ort.InferenceSession(model_path)

    def preprocess(self, frame):
        img = cv2.resize(frame, (640, 640))
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        return np.expand_dims(img, axis=0)

    def infer(self, frame):
        input_tensor = self.preprocess(frame)

        outputs = self.session.run(
            None,
            {self.session.get_inputs()[0].name: input_tensor}
        )

        return outputs[0]