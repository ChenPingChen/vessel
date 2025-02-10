import numpy as np
from ultralytics.models.yolo.detect.predict import DetectionPredictor

class ObjectDetect:
    def __init__(self, ckpt:str, conf: float=0.5):
        self.model = DetectionPredictor(overrides=dict(model=ckpt, conf=conf, source="", save=False, verbose=False))
        self.names = None
        
    def detect(self, image: np.ndarray):
        self.model.predict_cli(source=image)
        results = self.model.results
        if isinstance(self.names, type(None)):
            self.names = results[0].names
        return results[0].boxes.data