import ultralytics
from ultralytics import YOLO
import torch
from typing import Any

class DetectionModel():
    def __init__(self):
        ultralytics.checks()
        model_dir = "./models/best_yolov8_visdrone207212.pt"
        self.model = YOLO(model_dir)

    def process_frame(self, frame) -> None:
        result = self.model.track(source=frame, conf=0.5, iou=0.5,
                                       tracker="bytetrack.yaml",
                                       verbose=False, persist=True)
        self.boxes = result[0].boxes

    def get_coordinate(self) -> Any:
        # Returns the bounding boxes (x, y, height, width next frame)
        return [box.xywh.numpy() for box in self.boxes]
    
    def get_id(self) -> Any:
        # Returns the object IDs (next frame)
        return self.boxes.id[:]

    def get_confidence(self) -> Any:
        # Returns the object confidence (next frame)
        return self.boxes.conf[:]

    def get_class(self) -> Any:
        # Returns the object classes (next frame)
        return self.boxes.cls[:]