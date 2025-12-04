# utils.py
import cv2

def draw_annotations(frame, tracks, results):
    for trk in tracks:
        x1, y1, x2, y2 = map(int, trk.bbox)
        tid = trk.track_id

        use_phone = False
        for r in results:
            if r["track_id"] == tid:
                use_phone = r["use_phone"]

        color = (0, 255, 0) if not use_phone else (0, 0, 255)
        txt = f"ID {tid} {'PHONE' if use_phone else ''}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, txt, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Count people using phone
    phone_count = sum(1 for r in results if r["use_phone"])
    
    # Display counter
    cv2.putText(frame, f"Phone Users: {phone_count}", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    return frame
