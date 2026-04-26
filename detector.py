import torch
from ultralytics import YOLO
import cv2
import os

def load_model(model_path='best.pt'):
    model = YOLO(model_path)
    return model

def detect_players(video_path, model, output_dir="output_frames", conf_thresh=0.4, classes=None):
    cap = cv2.VideoCapture(video_path)
    frame_data = []
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    os.makedirs(f"{output_dir}/{video_name}", exist_ok=True)

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, verbose=False)[0]
        detections = []
        for box in results.boxes:
            cls = int(box.cls)
            conf = float(box.conf)
            if conf < conf_thresh:
                continue
            if classes is not None and cls not in classes:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            crop = frame[y1:y2, x1:x2]
            crop_path = f"{output_dir}/{video_name}/frame{frame_idx}_player{len(detections)}.jpg"
            cv2.imwrite(crop_path, crop)
            detections.append({
                "bbox": (x1, y1, x2, y2),
                "crop_path": crop_path,
                "class": cls,
                "confidence": conf
            })

        frame_data.append({
            "frame_idx": frame_idx,
            "detections": detections
        })
        frame_idx += 1

    cap.release()
    return frame_data
