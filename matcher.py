
import numpy as np
from scipy.spatial.distance import cosine

def match_players(frame_data_a, frame_data_b, similarity_threshold=0.5):
    matched_frames = []

    for frame_a, frame_b in zip(frame_data_a, frame_data_b):
        detections_a = frame_a['detections']
        detections_b = frame_b['detections']
        matches = []

        used_b = set()
        for i, det_a in enumerate(detections_a):
            best_match = -1
            best_score = float('inf')

            for j, det_b in enumerate(detections_b):
                if j in used_b:
                    continue

                sim = cosine(det_a['embedding'], det_b['embedding'])
                if sim < best_score:
                    best_score = sim
                    best_match = j

            if best_match != -1 and best_score < similarity_threshold:
                used_b.add(best_match)
                matches.append((i, best_match))

        matched_frames.append({
            "frame_idx": frame_a['frame_idx'],
            "matches": matches,
            "detections_a": detections_a,
            "detections_b": detections_b
        })

    return matched_frames
