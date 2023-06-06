from autodistill.detection import DetectionTargetModel
import supervision as sv
from ultralytics import YOLO

class YOLOv5(DetectionTargetModel):
    def __init__(self, model_name):
        self.yolo = YOLO(model_name)

    def predict(self, input:str, confidence=0.5) -> sv.Detections:
        pass

    def train(self, dataset_yaml, epochs=300):
        pass