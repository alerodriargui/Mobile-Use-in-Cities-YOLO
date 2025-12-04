import cv2
from detector import YOLODetector
from tracker import Tracker
from postprocess import MobileUsePostProcessor
from utils import draw_annotations
from yt_dlp import YoutubeDL   # <--- añadido

url = "https://www.youtube.com/watch?v=9NzUkgfpe9s"

def get_live_stream(url):
    ydl_opts = {
        'quiet': True,
        'skip_download': True,
        'format': 'best'
    }

    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        return info['url']

stream_url = get_live_stream(url)
cap = cv2.VideoCapture(stream_url)

# Inicializa detector, tracker y postprocess
detector = YOLODetector(model_path='yolov8n.pt', classes=['person'])
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
