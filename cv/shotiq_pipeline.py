"""
ShotIQ Stage 1 CV pipeline.

This is the first backend analysis engine for iPhone-recorded basketball workouts.
It is designed to run offline on uploaded video first, then later be upgraded for live use.

Core idea:
- Use pose estimation for shooter mechanics.
- Use ball detection/tracking for release and trajectory.
- Use manually pinned or detected hoop position for make/miss inference.

Run:
    python cv/shotiq_pipeline.py --video ./sample.mp4 --hoop-x 960 --hoop-y 260 --rim-width 120

Output:
    analysis.json with shot events and form metrics.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    import mediapipe as mp
except ImportError:  # pragma: no cover
    mp = None

Point = Tuple[float, float]


@dataclass
class HoopConfig:
    x: float
    y: float
    rim_width: float = 120.0

    @property
    def left(self) -> float:
        return self.x - self.rim_width / 2

    @property
    def right(self) -> float:
        return self.x + self.rim_width / 2

    @property
    def top(self) -> float:
        return self.y - self.rim_width * 0.12

    @property
    def bottom(self) -> float:
        return self.y + self.rim_width * 0.28


@dataclass
class BallObservation:
    frame: int
    time: float
    x: float
    y: float
    radius: float
    confidence: float


@dataclass
class PoseObservation:
    frame: int
    time: float
    wrist: Optional[Point]
    elbow: Optional[Point]
    shoulder: Optional[Point]
    hip: Optional[Point]
    knee: Optional[Point]


@dataclass
class ShotEvent:
    frame: int
    time: float
    release_x: float
    release_y: float
    arc_peak_y: float
    made: Optional[bool]
    confidence: float
    form_score: int
    notes: List[str]


class SimpleBallDetector:
    """HSV basketball detector.

    This is intentionally simple for Stage 1. It gives us a working baseline.
    Stage 2 should replace this with a YOLO/RF-DETR ball model.
    """

    def __init__(self) -> None:
        self.lower_orange = np.array([3, 60, 50])
        self.upper_orange = np.array([28, 255, 255])

    def detect(self, frame: np.ndarray, frame_idx: int, fps: float) -> Optional[BallObservation]:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_orange, self.upper_orange)
        mask = cv2.medianBlur(mask, 5)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        candidates = []
        h, w = frame.shape[:2]
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 20 or area > (w * h * 0.02):
                continue
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            if radius < 3 or radius > 80:
                continue
            circularity = area / (math.pi * radius * radius + 1e-6)
            if circularity < 0.25:
                continue
            candidates.append((area * circularity, x, y, radius, circularity))

        if not candidates:
            return None

        candidates.sort(reverse=True)
        _, x, y, radius, circularity = candidates[0]
        return BallObservation(
            frame=frame_idx,
            time=frame_idx / fps,
            x=float(x),
            y=float(y),
            radius=float(radius),
            confidence=float(min(1.0, circularity)),
        )


class PoseEstimator:
    def __init__(self, shooting_hand: str = "right") -> None:
        if mp is None:
            raise RuntimeError("mediapipe is not installed. Run pip install -r cv/requirements.txt")
        self.shooting_hand = shooting_hand
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.45,
            min_tracking_confidence=0.45,
        )

    def estimate(self, frame: np.ndarray, frame_idx: int, fps: float) -> PoseObservation:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self.pose.process(rgb)
        h, w = frame.shape[:2]

        def landmark(name: str) -> Optional[Point]:
            if not res.pose_landmarks:
                return None
            lm_id = getattr(self.mp_pose.PoseLandmark, name).value
            lm = res.pose_landmarks.landmark[lm_id]
            if lm.visibility < 0.35:
                return None
            return (lm.x * w, lm.y * h)

        side = "RIGHT" if self.shooting_hand.lower().startswith("r") else "LEFT"
        return PoseObservation(
            frame=frame_idx,
            time=frame_idx / fps,
            wrist=landmark(f"{side}_WRIST"),
            elbow=landmark(f"{side}_ELBOW"),
            shoulder=landmark(f"{side}_SHOULDER"),
            hip=landmark(f"{side}_HIP"),
            knee=landmark(f"{side}_KNEE"),
        )


def distance(a: Optional[Point], b: Optional[Point]) -> Optional[float]:
    if a is None or b is None:
        return None
    return math.hypot(a[0] - b[0], a[1] - b[1])


def form_score(pose: PoseObservation) -> Tuple[int, List[str]]:
    notes: List[str] = []
    score = 100

    if not pose.wrist or not pose.elbow or not pose.shoulder:
        return 55, ["Pose partially hidden. Keep full shooting arm visible."]

    wrist_x, wrist_y = pose.wrist
    elbow_x, elbow_y = pose.elbow
    shoulder_x, shoulder_y = pose.shoulder

    elbow_alignment = abs(wrist_x - elbow_x)
    shoulder_width_proxy = max(40.0, abs(shoulder_x - elbow_x) * 2.5)
    if elbow_alignment > shoulder_width_proxy * 0.55:
        score -= 22
        notes.append("Elbow drifts away from wrist line.")

    if wrist_y > shoulder_y:
        score -= 24
        notes.append("Release looks low. Get wrist above shoulder at release.")

    if elbow_y > shoulder_y + 70:
        score -= 12
        notes.append("Elbow is low before release.")

    return max(0, min(100, score)), notes


class ShotDetector:
    def __init__(self, hoop: HoopConfig) -> None:
        self.hoop = hoop
        self.ball_history: List[BallObservation] = []
        self.last_release_frame = -9999

    def update(self, ball: Optional[BallObservation], pose: PoseObservation) -> Optional[ShotEvent]:
        if ball:
            self.ball_history.append(ball)
            self.ball_history = self.ball_history[-45:]

        if not ball or not pose.wrist:
            return None

        wrist_ball_dist = distance(pose.wrist, (ball.x, ball.y))
        if wrist_ball_dist is None:
            return None

        recent = self.ball_history[-10:]
        if len(recent) < 5:
            return None

        # Release heuristic: ball was near wrist, then moves upward/away quickly.
        near_wrist = wrist_ball_dist < max(55, ball.radius * 5)
        ys = [b.y for b in recent]
        upward_motion = ys[-1] < ys[0] - 18
        enough_gap = ball.frame - self.last_release_frame > 35

        if not (near_wrist and upward_motion and enough_gap):
            return None

        score, notes = form_score(pose)
        made, conf = self._estimate_make()
        peak_y = min([b.y for b in self.ball_history[-30:]], default=ball.y)

        self.last_release_frame = ball.frame
        return ShotEvent(
            frame=ball.frame,
            time=ball.time,
            release_x=ball.x,
            release_y=ball.y,
            arc_peak_y=float(peak_y),
            made=made,
            confidence=conf,
            form_score=score,
            notes=notes,
        )

    def _estimate_make(self) -> Tuple[Optional[bool], float]:
        # Looks at recent ball path crossing the hoop region downward.
        if len(self.ball_history) < 8:
            return None, 0.0

        recent = self.ball_history[-30:]
        for prev, cur in zip(recent, recent[1:]):
            descending = cur.y > prev.y
            crosses_rim_y = prev.y < self.hoop.y <= cur.y or self.hoop.top <= cur.y <= self.hoop.bottom
            inside_rim_x = self.hoop.left <= cur.x <= self.hoop.right
            if descending and crosses_rim_y and inside_rim_x:
                return True, 0.62

        # If ball clearly descends beside the rim, likely miss, but keep lower confidence.
        for prev, cur in zip(recent, recent[1:]):
            descending = cur.y > prev.y
            near_rim_y = self.hoop.top <= cur.y <= self.hoop.bottom + 80
            beside_rim = abs(cur.x - self.hoop.x) < self.hoop.rim_width * 1.7
            outside_rim_x = not (self.hoop.left <= cur.x <= self.hoop.right)
            if descending and near_rim_y and beside_rim and outside_rim_x:
                return False, 0.45

        return None, 0.0


def analyze_video(video_path: Path, hoop: HoopConfig, shooting_hand: str = "right") -> Dict:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    detector = SimpleBallDetector()
    pose_estimator = PoseEstimator(shooting_hand=shooting_hand)
    shot_detector = ShotDetector(hoop=hoop)

    shots: List[ShotEvent] = []
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # Downscale for speed while preserving coordinates in scaled frame.
        max_width = 1280
        if frame.shape[1] > max_width:
            scale = max_width / frame.shape[1]
            frame = cv2.resize(frame, None, fx=scale, fy=scale)

        ball = detector.detect(frame, frame_idx, fps)
        pose = pose_estimator.estimate(frame, frame_idx, fps)
        event = shot_detector.update(ball, pose)
        if event:
            shots.append(event)

        frame_idx += 1

    cap.release()

    made = sum(1 for s in shots if s.made is True)
    missed = sum(1 for s in shots if s.made is False)
    known = made + missed
    avg_form = round(sum(s.form_score for s in shots) / len(shots), 1) if shots else 0

    return {
        "video": str(video_path),
        "fps": fps,
        "frames": frame_idx,
        "hoop": asdict(hoop),
        "summary": {
            "shots_detected": len(shots),
            "makes": made,
            "misses": missed,
            "unknown_results": len(shots) - known,
            "fg_pct": round((made / known) * 100, 1) if known else None,
            "average_form_score": avg_form,
        },
        "shots": [asdict(s) for s in shots],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--hoop-x", type=float, required=True)
    parser.add_argument("--hoop-y", type=float, required=True)
    parser.add_argument("--rim-width", type=float, default=120)
    parser.add_argument("--hand", choices=["right", "left"], default="right")
    parser.add_argument("--out", default="analysis.json")
    args = parser.parse_args()

    result = analyze_video(
        video_path=Path(args.video),
        hoop=HoopConfig(args.hoop_x, args.hoop_y, args.rim_width),
        shooting_hand=args.hand,
    )
    Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
