
import os
import cv2
import numpy as np
import pandas as pd

from collections import defaultdict
from ultralytics import YOLO
from sklearn.cluster import KMeans


# ==========================================
# 1. JERSEY COLOR EXTRACTION
# ==========================================

def extract_jersey_color(frame, box):

    x1, y1, x2, y2 = map(int, box)

    h, w = frame.shape[:2]

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)

    player_h = y2 - y1
    player_w = x2 - x1

    if player_h < 10 or player_w < 10:
        return None

    torso = frame[
        y1 + int(player_h * 0.15):
        y1 + int(player_h * 0.55),
        x1 + int(player_w * 0.20):
        x1 + int(player_w * 0.80)
    ]

    if torso.size == 0:
        return None

    hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(
        hsv,
        np.array([0, 45, 40]),
        np.array([180, 255, 255])
    )

    pixels = torso[mask > 0]

    if len(pixels) < 10:
        return None

    return np.median(pixels, axis=0)


# ==========================================
# 2. MAIN ANALYTICS FUNCTION
# ==========================================

def process_video(
    video_path,
    model_path,
    output_path,
    progress_callback=None,
    confidence=0.25
):

    os.makedirs("outputs", exist_ok=True)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    model = YOLO(model_path)

    # ======================================
    # FIRST PASS: COLLECT PLAYER COLORS
    # ======================================

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError("Video open nahi hui!")

    track_colors = defaultdict(list)

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    frame_no = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        results = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=confidence,
            verbose=False
        )

        result = results[0]

        if (
            result.boxes is not None
            and result.boxes.id is not None
        ):

            boxes = result.boxes.xyxy.cpu().numpy()
            ids = result.boxes.id.int().cpu().tolist()
            classes = result.boxes.cls.int().cpu().tolist()

            for box, track_id, cls in zip(
                boxes, ids, classes
            ):

                if cls != 0:
                    continue

                color = extract_jersey_color(
                    frame, box
                )

                if color is not None:
                    track_colors[track_id].append(color)

        frame_no += 1

        if progress_callback and total_frames > 0:
            progress_callback(
                0.45 * frame_no / total_frames
            )

    cap.release()

    # ======================================
    # THIRD: AUTOMATIC TEAM CLASSIFICATION
    # ======================================

    player_features = []
    player_ids = []

    for track_id, colors in track_colors.items():

        if len(colors) < 2:
            continue

        average_color = np.median(
            colors, axis=0
        )

        player_features.append(average_color)
        player_ids.append(track_id)

    if len(player_features) < 2:

        raise ValueError(
            "Two teams identify karne ke liye "
            "sufficient player color samples nahi mile."
        )

    X = np.array(player_features)

    kmeans = KMeans(
        n_clusters=2,
        random_state=42,
        n_init=10
    )

    labels = kmeans.fit_predict(X)

    player_teams = {
        track_id: int(label)
        for track_id, label in zip(player_ids, labels)
    }

    print("Automatic team classification completed!")

    for team_id in range(2):

        count = sum(
            team == team_id
            for team in player_teams.values()
        )

        print(
            f"Team {team_id + 1}: {count} tracked players"
        )

    # ======================================
    # SECOND PASS: VIDEO ANNOTATION
    # ======================================

    cap = cv2.VideoCapture(video_path)
    model = YOLO(model_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    fps = fps if fps > 0 else 25

    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    if not writer.isOpened():
        cap.release()
        raise ValueError("Output video writer open nahi hua!")

    player_records = []
    possession_counts = defaultdict(int)

    player_positions = defaultdict(list)

    ball_count = 0
    referee_count = 0
    frame_no = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        results = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=confidence,
            verbose=False
        )

        result = results[0]

        players_in_frame = []
        ball_center = None

        if result.boxes is not None:

            boxes = result.boxes.xyxy.cpu().numpy()
            classes = result.boxes.cls.int().cpu().tolist()

            if result.boxes.id is not None:
                ids = result.boxes.id.int().cpu().tolist()
            else:
                ids = [-1] * len(boxes)

            for box, track_id, cls in zip(
                boxes, ids, classes
            ):

                x1, y1, x2, y2 = map(int, box)

                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2

                # PLAYER
                if cls == 0:

                    team_id = player_teams.get(
                        track_id, -1
                    )

                    if team_id == 0:
                        color = (255, 100, 0)
                        team_name = "Team 1"

                    elif team_id == 1:
                        color = (0, 0, 255)
                        team_name = "Team 2"

                    else:
                        color = (200, 200, 200)
                        team_name = "Unknown"

                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        color,
                        2
                    )

                    cv2.putText(
                        frame,
                        f"ID {track_id} | {team_name}",
                        (x1, max(y1 - 8, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        2
                    )

                    if track_id >= 0:

                        players_in_frame.append(
                            (track_id, team_id, cx, cy)
                        )

                        player_positions[track_id].append(
                            (frame_no, cx, cy)
                        )

                        player_records.append({
                            "Frame": frame_no,
                            "Player_ID": track_id,
                            "Team": team_name,
                            "X": cx,
                            "Y": cy
                        })

                # BALL
                elif cls == 1:

                    ball_count += 1
                    ball_center = (cx, cy)

                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 255),
                        2
                    )

                    cv2.putText(
                        frame,
                        "BALL",
                        (x1, max(y1 - 8, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 255),
                        2
                    )

                # REFEREE
                elif cls == 2:

                    referee_count += 1

                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        (255, 255, 255),
                        2
                    )

                    cv2.putText(
                        frame,
                        "REFEREE",
                        (x1, max(y1 - 8, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        2
                    )

        # APPROXIMATE BALL POSSESSION

        if ball_center is not None and players_in_frame:

            distances = []

            for pid, team_id, px, py in players_in_frame:

                distance = np.sqrt(
                    (px - ball_center[0]) ** 2
                    + (py - ball_center[1]) ** 2
                )

                distances.append(
                    (distance, team_id)
                )

            nearest_distance, nearest_team = min(
                distances, key=lambda x: x[0]
            )

            if nearest_distance < 120 and nearest_team in [0, 1]:

                possession_counts[nearest_team] += 1

        writer.write(frame)

        frame_no += 1

        if progress_callback and total_frames > 0:

            progress_callback(
                0.45 + 0.55 * frame_no / total_frames
            )

    cap.release()
    writer.release()

    # ======================================
    # PLAYER MOVEMENT STATISTICS
    # ======================================

    movement_records = []

    for pid, positions in player_positions.items():

        positions.sort(key=lambda x: x[0])

        distance = 0

        for i in range(1, len(positions)):

            previous = positions[i - 1]
            current = positions[i]

            # Ignore gaps between tracking observations
            if current[0] - previous[0] > 2:
                continue

            dx = current[1] - previous[1]
            dy = current[2] - previous[2]

            distance += np.sqrt(dx ** 2 + dy ** 2)

        movement_records.append({
            "Player_ID": pid,
            "Team": (
                f"Team {player_teams[pid] + 1}"
                if pid in player_teams else "Unknown"
            ),
            "Movement_Pixels": round(distance, 2),
            "Frames_Tracked": len(positions)
        })

    players_df = pd.DataFrame(movement_records)

    tracking_df = pd.DataFrame(player_records)

    # ======================================
    # TEAM STATISTICS
    # ======================================

    total_possession = sum(possession_counts.values())

    team_records = []

    for team_id in range(2):

        team_players = sum(
            1 for team in player_teams.values()
            if team == team_id
        )

        possession = (
            possession_counts[team_id]
            / total_possession * 100
            if total_possession > 0 else 0
        )

        team_records.append({
            "Team": f"Team {team_id + 1}",
            "Unique_Players": team_players,
            "Possession_Percent": round(possession, 2)
        })

    teams_df = pd.DataFrame(team_records)

    # ======================================
    # SAVE CSV REPORTS
    # ======================================

    players_df.to_csv(
        "outputs/player_statistics.csv",
        index=False
    )

    teams_df.to_csv(
        "outputs/team_statistics.csv",
        index=False
    )

    tracking_df.to_csv(
        "outputs/player_tracking.csv",
        index=False
    )

    return {
        "players": players_df,
        "teams": teams_df,
        "tracking": tracking_df,
        "total_frames": frame_no,
        "ball_detections": ball_count,
        "referee_detections": referee_count,
        "output_video": output_path
    }
