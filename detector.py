# detector.py
from ultralytics import YOLO
import cv2

class YOLODetector:
    def __init__(self, model_path, classes):
        self.model = YOLO(model_path)
        self.classes = classes
        # Resolve class IDs
        self.class_ids = []
        for name in classes:
            for id, cls_name in self.model.names.items():
                if cls_name == name:
                    self.class_ids.append(id)
                    break

    def predict(self, frame):
        # Lower confidence threshold to detect small objects like phones
            # Filter classes during inference for speed and precision
        results = self.model(frame, conf=0.1, classes=self.class_ids)[0]

        detections = []
        for box in results.boxes:
            cls_id = int(box.cls)
            cls_name = self.model.names[cls_id]
            score = float(box.conf.cpu().numpy())

            # Class-specific thresholds
            if cls_name == 'person' and score < 0.5:
                continue
            if cls_name == 'cell phone' and score < 0.1:
                continue

            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

            detections.append({
                "bbox": [x1, y1, x2, y2],
                "score": score,
                "class": cls_name
            })

        return detections
