# detector.py
from ultralytics import YOLO
import cv2

class YOLODetector:
    def __init__(self, model_path, classes):
        self.model = YOLO(model_path)
        self.classes = classes

    def predict(self, frame):
        results = self.model(frame)[0]

        detections = []
        for box in results.boxes:
            cls_id = int(box.cls)
            cls_name = self.classes[cls_id] if cls_id < len(self.classes) else "unknown"
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            score = float(box.conf.cpu().numpy())

            detections.append({
                "bbox": [x1, y1, x2, y2],
                "score": score,
                "class": cls_name
            })

        return detections
