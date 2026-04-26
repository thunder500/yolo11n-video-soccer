"""Single-video pipeline: detect players, embed with ResNet, track with IOU + appearance, draw IDs."""

import colorsys
import os
from typing import Any

import cv2
import numpy as np
import torch
from scipy.spatial.distance import cosine

from detector import detect_players, load_model
from feature_extractor import extract_all_features, load_resnet_model

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IOU_THRESHOLD = 0.3
APPEARANCE_THRESHOLD = 0.45
MAX_TRACK_AGE_FRAMES = 30


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    a_area = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    b_area = max(0, bx2 - bx1) * max(0, by2 - by1)
    denom = a_area + b_area - inter
    return inter / denom if denom > 0 else 0.0


def _id_color(pid: int) -> tuple[int, int, int]:
    # stable HSV → BGR, well-separated hues
    h = (pid * 0.6180339887) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.85, 0.95)
    return (int(b * 255), int(g * 255), int(r * 255))


def assign_ids(frame_data: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    tracks: dict[int, dict[str, Any]] = {}
    next_id = 0

    for frame in frame_data:
        dets = frame["detections"]
        ids = [-1] * len(dets)
        used = set()

        # Pass 1 — IOU with recent tracks (handles the easy continuous-motion case)
        for i, det in enumerate(dets):
            best_tid, best_iou = -1, IOU_THRESHOLD
            for tid, t in tracks.items():
                if tid in used:
                    continue
                if frame["frame_idx"] - t["last_frame"] > 3:
                    continue
                v = _iou(det["bbox"], t["bbox"])
                if v > best_iou:
                    best_iou, best_tid = v, tid
            if best_tid >= 0:
                ids[i] = best_tid
                used.add(best_tid)

        # Pass 2 — appearance match (handles short occlusions)
        for i, det in enumerate(dets):
            if ids[i] >= 0:
                continue
            best_tid, best_dist = -1, APPEARANCE_THRESHOLD
            for tid, t in tracks.items():
                if tid in used:
                    continue
                if frame["frame_idx"] - t["last_frame"] > MAX_TRACK_AGE_FRAMES:
                    continue
                d = cosine(det["embedding"], t["embedding"])
                if d < best_dist:
                    best_dist, best_tid = d, tid
            if best_tid >= 0:
                ids[i] = best_tid
                used.add(best_tid)

        # Pass 3 — new IDs
        for i in range(len(dets)):
            if ids[i] < 0:
                ids[i] = next_id
                next_id += 1

        # Update tracks
        for i, det in enumerate(dets):
            tid = ids[i]
            tracks[tid] = {
                "bbox": det["bbox"],
                "embedding": det["embedding"],
                "last_frame": frame["frame_idx"],
            }

        frame["track_ids"] = ids

    return frame_data, next_id


def draw_and_export(video_path: str, frame_data: list[dict[str, Any]], output_path: str) -> None:
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    idx = 0
    try:
        while cap.isOpened():
            ret, img = cap.read()
            if not ret:
                break
            if idx < len(frame_data):
                frame = frame_data[idx]
                for i, det in enumerate(frame["detections"]):
                    x1, y1, x2, y2 = det["bbox"]
                    pid = frame["track_ids"][i]
                    color = _id_color(pid)
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                    label = f"ID {pid}"
                    (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(img, (x1, y1 - th - baseline - 4), (x1 + tw + 6, y1), color, -1)
                    cv2.putText(img, label, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (10, 10, 10), 2)
            writer.write(img)
            idx += 1
    finally:
        cap.release()
        writer.release()


def run(input_video: str = "input.mp4", output_video: str = "output.mp4") -> None:
    print("🔍 Loading models...", flush=True)
    if os.path.exists("best.pt"):
        yolo = load_model("best.pt")
        classes = None
    else:
        print("⚠️  best.pt not found — falling back to yolo11n.pt (COCO person class)", flush=True)
        yolo = load_model("yolo11n.pt")
        classes = [0]
    resnet = load_resnet_model().to(DEVICE)

    print("🧠 Running YOLO detection...", flush=True)
    frames = detect_players(input_video, yolo, output_dir="output_frames", classes=classes)

    print("📊 Extracting ResNet features...", flush=True)
    frames = extract_all_features(frames, resnet, device=DEVICE)

    print("🔗 Matching players across frames...", flush=True)
    frames, num_ids = assign_ids(frames)
    print(f"    {num_ids} unique player IDs assigned.", flush=True)

    print("🖼️ Visualizing tracks...", flush=True)
    draw_and_export(input_video, frames, output_video)

    print(f"🎞️ Saved video to {output_video}", flush=True)
    print("✅ Done!", flush=True)


if __name__ == "__main__":
    import sys
    inp = sys.argv[1] if len(sys.argv) > 1 else "input.mp4"
    outp = sys.argv[2] if len(sys.argv) > 2 else "output.mp4"
    run(inp, outp)
