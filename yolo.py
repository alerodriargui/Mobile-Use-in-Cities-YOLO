import cv2
from detector import YOLODetector
from tracker import Tracker
from postprocess import MobileUsePostProcessor
from utils import draw_annotations
import subprocess

url = "https://www.youtube.com/watch?v=9NzUkgfpe9s"

# Extrae la URL directa de streaming
def get_live_stream(url):
    cmd = ["yt-dlp", "-g", url]
    stream_url = subprocess.check_output(cmd).decode().strip()
    return stream_url

stream_url = get_live_stream(url)

# Ahora sí OpenCV puede abrir la URL directa
cap = cv2.VideoCapture(stream_url)

# Inicializa detector, tracker y postprocess
detector = YOLODetector(model_path='models/best_phone_model.pt', classes=['person','phone','hand'])
tracker  = Tracker()
post = MobileUsePostProcessor(history_len=30, fps=25)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    detections = detector.predict(frame)
    tracks = tracker.update(detections, frame)
    results = post.update(tracks, detections, frame_time=cap.get(cv2.CAP_PROP_POS_FRAMES))
    vis = draw_annotations(frame, tracks, results)
    cv2.imshow("MobileUse Live", vis)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
