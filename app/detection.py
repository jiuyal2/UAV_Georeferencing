import ultralytics
from ultralytics import YOLO
import torch
from typing import Any

class DetectionModel():
    def __init__(self, vid_in_path):
        ultralytics.checks()
        model_dir = "../models/best_yolov8_visdrone207212.pt"
        self.model = YOLO(model_dir)
        self.results = iter(self.model.track(source=vid_in_path, conf=0.5, iou=0.5, tracker="bytetrack.yaml", stream=True, verbose=False))

    def next_frame(self) -> None:
        self.result = next(self.results)
        self.boxes = self.result.boxes

    def get_coordinate(self) -> Any:
        # Returns the bounding boxes (x, y, height, width next frame)
        return [box.xywh for box in self.boxes]
    
    def get_id(self) -> Any:
        # Returns the object IDs (next frame)
        return self.boxes.id[:]

    def get_confidence(self) -> Any:
        # Returns the object confidence (next frame)
        return self.boxes.conf[:]

    def get_class(self) -> Any:
        # Returns the object classes (next frame)
        return self.boxes.cls[:]