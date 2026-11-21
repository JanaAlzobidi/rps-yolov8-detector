"""Shared helpers for the Rock Paper Scissors detector.

Used by both app.py (Gradio web app) and predict.py (command-line testing),
so the game rules and drawing live in one place.
"""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

DEFAULT_WEIGHTS = Path(__file__).parent / "weights" / "best.pt"

# What each gesture beats.
BEATS = {"rock": "scissors", "scissors": "paper", "paper": "rock"}


@dataclass
class Hand:
    gesture: str      # "rock", "paper" or "scissors"
    confidence: float
    x_center: float   # horizontal centre of the box, in pixels
    box_index: int    # position of this box in the YOLO result


@dataclass
class Round:
    hands: list       # every detected hand, left to right
    player1: Hand | None
    player2: Hand | None
    verdict: str      # human-readable result


def load_model(weights=DEFAULT_WEIGHTS):
    """Load the trained YOLOv8 model and check its class names."""
    weights = Path(weights)
    if not weights.exists():
        raise FileNotFoundError(
            f"Model not found: {weights}\n"
            "Download best.pt from the training notebook and put it in the weights/ folder, "
            "or pass its path with --weights."
        )
    model = YOLO(str(weights))
    names = {n.lower() for n in model.names.values()}
    unknown = names - set(BEATS)
    if unknown:
        print(f"Warning: the model has classes the game does not know: {sorted(unknown)}")
    return model


def model_imgsz(model, default=640):
    """Image size the model was trained with (falls back to 640)."""
    try:
        return int(model.overrides.get("imgsz") or default)
    except (TypeError, ValueError):
        return default


def hands_from_result(result):
    """Turn a YOLO result into a list of Hand objects, sorted left to right."""
    boxes = result.boxes
    hands = [
        Hand(
            gesture=result.names[int(boxes.cls[i])].lower(),
            confidence=float(boxes.conf[i]),
            x_center=float(boxes.xywh[i][0]),
            box_index=i,
        )
        for i in range(len(boxes))
    ]
    return sorted(hands, key=lambda h: h.x_center)


def judge(hands):
    """Decide the round.

    The two most confident hands are the players. The one further left in the
    image is Player 1, the one further right is Player 2.
    """
    if len(hands) < 2:
        found = "no hands" if not hands else f"1 hand ({hands[0].gesture})"
        return Round(hands, None, None, f"Need 2 hands, found {found}")

    top2 = sorted(hands, key=lambda h: h.confidence, reverse=True)[:2]
    p1, p2 = sorted(top2, key=lambda h: h.x_center)

    if p1.gesture == p2.gesture:
        verdict = f"Draw: both played {p1.gesture}"
    elif BEATS.get(p1.gesture) == p2.gesture:
        verdict = f"Player 1 (left) wins: {p1.gesture} beats {p2.gesture}"
    elif BEATS.get(p2.gesture) == p1.gesture:
        verdict = f"Player 2 (right) wins: {p2.gesture} beats {p1.gesture}"
    else:
        verdict = f"Unknown gestures: {p1.gesture} vs {p2.gesture}"
    return Round(hands, p1, p2, verdict)


def play(model, image, conf=0.5, imgsz=None):
    """Run the model on one image and judge the round.

    `image` can be a file path, a PIL image (RGB) or a numpy array in BGR order
    (as read by OpenCV). Returns (annotated BGR image, Round).
    """
    # agnostic_nms: one box per hand. Without it, the same hand can get two boxes
    # with different classes (e.g. rock 0.5 and scissors 0.7) and count as two players.
    result = model.predict(
        image, imgsz=imgsz or model_imgsz(model), conf=conf, agnostic_nms=True, verbose=False
    )[0]
    rnd = judge(hands_from_result(result))
    annotated = result.plot()
    if rnd.player1 and rnd.player2:
        annotated = label_players(annotated, result, rnd)
    return annotated, rnd


def label_players(frame, result, rnd):
    """Write "P1" / "P2" under the two player boxes."""
    scale = max(frame.shape[:2]) / 700
    for tag, hand in (("P1", rnd.player1), ("P2", rnd.player2)):
        x1, _, _, y2 = (int(v) for v in result.boxes.xyxy[hand.box_index])
        draw_text(frame, tag, (x1 + 6, y2 - 10), scale)
    return frame


def draw_text(frame, text, org, scale=1.0):
    """White text with a black outline, readable on any background."""
    scale = max(scale, 0.5)
    thick = max(int(2 * scale), 1)
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick * 3, cv2.LINE_AA)
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thick, cv2.LINE_AA)
    return frame


def add_verdict_banner(frame, verdict):
    """Write the verdict in the top-left corner of a BGR frame."""
    scale = max(frame.shape[:2]) / 800
    return draw_text(frame, verdict, (int(20 * scale) + 5, int(50 * scale) + 10), scale)


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(image[..., ::-1])
