---
title: Soccer Player Tracker
emoji: ⚽
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# Soccer Player Tracker

Upload a soccer match clip and the app will detect every player, give each one a stable colored ID, and keep that ID consistent across every frame.

## How it works

1. **Detect** — YOLO finds players in each frame
2. **Fingerprint** — ResNet50 turns each player crop into a feature vector
3. **Track** — IOU + appearance matching links the same player across frames
4. **Render** — annotated MP4 with colored boxes + ID numbers per player

## Run locally

```bash
git clone https://github.com/thunder500/yolo11n-video-soccer.git
cd yolo11n-video-soccer
python -m venv .venv
.venv/Scripts/activate          # Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Then open http://localhost:5000.

If you have a fine-tuned `best.pt`, drop it in the project root — otherwise the pipeline auto-downloads `yolo11n.pt` and filters to the COCO `person` class.

## Run with Docker

```bash
docker build -t soccer-tracker .
docker run -p 7860:7860 soccer-tracker
```

## Tech

YOLO · ResNet50 · OpenCV · PyTorch · Flask
