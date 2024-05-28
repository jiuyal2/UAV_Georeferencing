import ultralytics
from ultralytics import YOLO
import numpy as np
from typing import Any
from pathlib import Path

class DetectionModel():
    def __init__(self):
        """
        Wrapper class for YOLOv8's tracking model, with frame-by-frame tracking.
        Call process_frame() to process the first/next input frame. Convenince methods are provided.
        """
        ultralytics.checks()
        # model_dir = "./src/detector/best_yolov9.pt"
        path = Path(__file__).parent
        model_dir = path / "./src/detector/v9_best_may27.pt"
        self.model = YOLO(model_dir)

    def process_frame(self, frame) -> None:
        result = self.model.track(source=frame, conf=0.5, iou=0.5,
                                       tracker="bytetrack.yaml",
                                       device = [0, 1],
                                       verbose=False, persist=True)
        self.boxes = result[0].boxes

    def get_coordinate(self) -> list[np.ndarray]:
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