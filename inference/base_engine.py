from abc import ABC, abstractmethod

class BaseEngine(ABC):

    @abstractmethod
    def infer(self, frame):
        """Run model inference on a single frame"""
        pass