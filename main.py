
import torch
from detector import load_model, detect_players
from feature_extractor import load_resnet_model, extract_all_features
from matcher import match_players
import cv2
import os
import glob

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def draw_matches(matched_frame, out_dir, frame_source):
    idx = matched_frame['frame_idx']
    matches = matched_frame['matches']
    dets_a = matched_frame['detections_a']
    dets_b = matched_frame['detections_b']

    frame_a_path = f"output_frames/{frame_source[0]}/frame{idx}_full.jpg"
    frame_b_path = f"output_frames/{frame_source[1]}/frame{idx}_full.jpg"
    frame_a = cv2.imread(frame_a_path)
    frame_b = cv2.imread(frame_b_path)

    for det in dets_a:
        x1, y1, x2, y2 = det['bbox']
        cv2.rectangle(frame_a, (x1, y1), (x2, y2), (128, 128, 128), 1)

    for det in dets_b:
        x1, y1, x2, y2 = det['bbox']
        cv2.rectangle(frame_b, (x1, y1), (x2, y2), (128, 128, 128), 1)

    for pid, (i, j) in enumerate(matches):
        xa1, ya1, xa2, ya2 = dets_a[i]['bbox']
        xb1, yb1, xb2, yb2 = dets_b[j]['bbox']

        cv2.rectangle(frame_a, (xa1, ya1), (xa2, ya2), (0, 255, 0), 2)
        cv2.putText(frame_a, f"ID {pid}", (xa1, ya1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.rectangle(frame_b, (xb1, yb1), (xb2, yb2), (255, 0, 0), 2)
        cv2.putText(frame_b, f"ID {pid}", (xb1, yb1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    os.makedirs(out_dir, exist_ok=True)
    cv2.imwrite(f"{out_dir}/broadcast_frame{idx}.jpg", frame_a)
    cv2.imwrite(f"{out_dir}/tacticam_frame{idx}.jpg", frame_b)

def extract_full_frames(video_path, save_dir):
    cap = cv2.VideoCapture(video_path)
    name = os.path.splitext(os.path.basename(video_path))[0]
    os.makedirs(f"{save_dir}/{name}", exist_ok=True)
    idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(f"{save_dir}/{name}/frame{idx}_full.jpg", frame)
        idx += 1
    cap.release()

def frames_to_video(input_folder, output_path, prefix="broadcast", fps=30):
    images = sorted(glob.glob(f"{input_folder}/{prefix}_frame*.jpg"),
                    key=lambda x: int(x.split("frame")[-1].split(".")[0]))
    
    if not images:
        print(f"No frames found for prefix '{prefix}' in {input_folder}")
        return

    frame = cv2.imread(images[0])
    height, width, _ = frame.shape
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    for img_path in images:
        frame = cv2.imread(img_path)
        out.write(frame)

    out.release()
    print(f"🎞️ Saved video to {output_path}")

if __name__ == "__main__":
    print("🔍 Loading models...")
    if os.path.exists("best.pt"):
        yolo_model = load_model("best.pt")
        det_classes = None
    else:
        print("⚠️  best.pt not found — falling back to yolo11n.pt (COCO person class)")
        yolo_model = load_model("yolo11n.pt")
        det_classes = [0]
    resnet_model = load_resnet_model().to(DEVICE)

    print("📼 Extracting frames from videos...")
    extract_full_frames("broadcast.mp4", "output_frames")
    extract_full_frames("tacticam.mp4", "output_frames")

    print("🧠 Running YOLO detection...")
    broadcast_frames = detect_players("broadcast.mp4", yolo_model, output_dir="output_frames", classes=det_classes)
    tacticam_frames = detect_players("tacticam.mp4", yolo_model, output_dir="output_frames", classes=det_classes)

    print("📊 Extracting ResNet features...")
    broadcast_frames = extract_all_features(broadcast_frames, resnet_model, device=DEVICE)
    tacticam_frames = extract_all_features(tacticam_frames, resnet_model, device=DEVICE)

    print("🔗 Matching players across views...")
    matched_data = match_players(broadcast_frames, tacticam_frames)

    print("🖼️ Visualizing matched players...")
    for frame in matched_data:
        draw_matches(frame, "output_matches", frame_source=("broadcast", "tacticam"))

    print("✅ Done! Matched frames saved in 'output_matches/'")
    frames_to_video("output_matches", "broadcast_output.mp4", prefix="broadcast")
    frames_to_video("output_matches", "tacticam_output.mp4", prefix="tacticam")

