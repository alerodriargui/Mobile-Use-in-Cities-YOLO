# postprocess.py
class MobileUsePostProcessor:
    def __init__(self, history_len=30, fps=25):
        self.history_len = history_len
        self.fps = fps
        self.history = {}  # track_id → detections history

    def update(self, tracks, detections, frame_time):
        results = []

        for trk in tracks:
            tid = trk.track_id
            if tid not in self.history:
                self.history[tid] = []

            # detect if phone near person:
            has_phone = False
            for det in detections:
                if det["class"] == "phone":
                    phone = det["bbox"]
                    person = trk.bbox

                    px = (person[0] + person[2]) / 2
                    py = (person[1] + person[3]) / 2
                    cx = (phone[0] + phone[2]) / 2
                    cy = (phone[1] + phone[3]) / 2

                    dist = ((px - cx)**2 + (py - cy)**2)**0.5
                    if dist < 200:
                        has_phone = True

            self.history[tid].append(has_phone)
            if len(self.history[tid]) > self.history_len:
                self.history[tid].pop(0)

            # decision rule
            use_phone = sum(self.history[tid]) > self.history_len * 0.4

            results.append({
                "track_id": tid,
                "use_phone": use_phone
            })

        return results
