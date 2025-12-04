# tracker.py
import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment

def iou(b1, b2):
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)

    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])

    return inter / (a1 + a2 - inter + 1e-6)

class Track:
    def __init__(self, bbox, track_id):
        self.bbox = bbox
        self.track_id = track_id
        self.age = 0
        self.hits = 1

class Tracker:
    def __init__(self, iou_threshold=0.3):
        self.iou_threshold = iou_threshold
        self.tracks = []
        self.next_id = 0

    def update(self, detections, frame):
        det_bboxes = [d["bbox"] for d in detections]
        det_classes = [d["class"] for d in detections]

        if len(self.tracks) == 0:
            for bbox in det_bboxes:
                self.tracks.append(Track(bbox, self.next_id))
                self.next_id += 1
            return self.tracks

        # IOU matrix
        iou_matrix = np.zeros((len(det_bboxes), len(self.tracks)))
        for i, det in enumerate(det_bboxes):
            for j, trk in enumerate(self.tracks):
                iou_matrix[i, j] = iou(det, trk.bbox)

        row_inds, col_inds = linear_sum_assignment(-iou_matrix)

        used_tracks = set()
        for row, col in zip(row_inds, col_inds):
            if iou_matrix[row, col] > self.iou_threshold:
                self.tracks[col].bbox = det_bboxes[row]
                self.tracks[col].age = 0
                self.tracks[col].hits += 1
                used_tracks.add(col)

        # Add new tracks
        for i, bbox in enumerate(det_bboxes):
            if i not in row_inds:
                self.tracks.append(Track(bbox, self.next_id))
                self.next_id += 1

        # Age tracks
        self.tracks = [t for t in self.tracks if t.age < 10]
        for t in self.tracks:
            t.age += 1

        return self.tracks
