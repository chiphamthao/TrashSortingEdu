"""
Part 2: Multi-item pinch & sort. Try to sort as many correctly as possible before time runs out
"""

import cv2 as cv
import numpy as np
import time
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional
import mediapipe as mp
from utils.draw_banner import draw_banner
from utils.norm_to_px import norm_to_px
from utils.load_item import load_item
from utils.draw_text_centered import draw_text_centered

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

WINDOW_NAME = "Pinch & Sort"
FONT = cv.FONT_HERSHEY_SIMPLEX

BG = (25, 25, 30)
FG = (240, 240, 240)
ACCENT = (128, 64, 255)

ITEM_RADIUS = 30
GRAB_RADIUS = 40
BIN_HEIGHT = 120
BIN_GAP = 12
TOP_BANNER = 80

GAME_TIME_LIMIT = 60.0  
NUM_ITEMS_ON_SCREEN = 10 

# landmark ids
THUMB_TIP = 4
INDEX_TIP = 8
INDEX_MCP = 5
PINKY_MCP = 17

@dataclass
class Item:
    name: str
    correct_bin: str

@dataclass
class ActiveItem:
    item: Item
    pos: Tuple[int, int]
    active: bool = True

ITEMS: List[Item] = [
    Item("Banana", "Compost"),
    Item("Can With Food", "Landfill"),
    Item("Coffee Grounds", "Compost"),
    Item("Detergent", "Recycling"),
    Item("Egg Shells", "Compost"),
    Item("Plastic Fork", "Landfill"),
    Item("Magazine", "Recycling"),
    Item("Plastic Bag", "Landfill"),
    Item("Plastic Container", "Recycling"),
    Item("Styrofoam", "Landfill"),
]

BINS = [
    ("Recycling", (40, 140, 255)),
    ("Compost",   (60, 200, 60)),
    ("Landfill",  (80, 80, 80)),
]

ITEM_FILES = {
    "Banana": "banana.png",
    "Can With Food": "canWithFood.png",
    "Coffee Grounds": "coffeeGround.png",
    "Detergent": "detergent.png",
    "Egg Shells": "eggShells.png",
    "Plastic Fork": "fork.png",
    "Magazine": "magazine.png",
    "Plastic Bag": "plasticBag.png",
    "Plastic Container": "plasticContainer.png",
    "Styrofoam": "styrofoam.png",
}

ITEM_IMAGES = {k: load_item(f"data/{v}", ITEM_RADIUS) for k, v in ITEM_FILES.items()}

def l2(a, b):
    return float(np.linalg.norm(np.array(a, dtype=np.float32) - np.array(b, dtype=np.float32)))

def pinch_state(landmarks_px, bounds, was_pinched):
    p_thumb = landmarks_px[THUMB_TIP]
    p_index = landmarks_px[INDEX_TIP]
    dist = l2(p_thumb, p_index)
    span = l2(landmarks_px[INDEX_MCP], landmarks_px[PINKY_MCP])

    on_t = bounds[0] * span
    off_t = bounds[1] * span

    if was_pinched:
        return dist < off_t
    else:
        return dist < on_t

def bin_hit(pos, bins) -> Optional[str]:
    x, y = pos
    R = 2 * ITEM_RADIUS

    for label, _, (x1, y1, x2, y2) in bins:
        closest_x = min(max(x, x1), x2)
        closest_y = min(max(y, y1), y2)
        dx = x - closest_x
        dy = y - closest_y
        if dx * dx + dy * dy <= R * R:
            return label

    return None

def final_screen(frame, score_correct):
    h, w = frame.shape[:2]
    cv.rectangle(frame, (0, 0), (w, h), (30, 30, 30), -1)
    cv.putText(frame, "Round over!", (40, 150), FONT, 1.6, (255, 255, 255), 3, cv.LINE_AA)
    cv.putText(frame, f"Correctly sorted: {score_correct}/10", (40, 230),
               FONT, 1.2, (120, 255, 120), 3, cv.LINE_AA)
    cv.putText(frame, "Press 'q' to exit", (40, 290), FONT, 0.9, (220, 220, 220), 2, cv.LINE_AA)

def overlay_at_center(frame, overlay, center):
    if overlay is None or overlay.shape[2] < 4:
        return

    h, w = frame.shape[:2]
    oh, ow = overlay.shape[:2]
    cx, cy = center

    x1 = int(cx - ow / 2)
    y1 = int(cy - oh / 2)
    x2 = x1 + ow
    y2 = y1 + oh

    x1_clamp = max(0, x1)
    y1_clamp = max(TOP_BANNER, y1)
    x2_clamp = min(w, x2)
    y2_clamp = min(h, y2)

    if x1_clamp >= x2_clamp or y1_clamp >= y2_clamp:
        return

    ox1 = x1_clamp - x1
    oy1 = y1_clamp - y1
    ox2 = ox1 + (x2_clamp - x1_clamp)
    oy2 = oy1 + (y2_clamp - y1_clamp)

    roi = frame[y1_clamp:y2_clamp, x1_clamp:x2_clamp]
    ov = overlay[oy1:oy2, ox1:ox2]

    alpha = ov[:, :, 3] / 255.0
    alpha = alpha[..., None]
    rgb = ov[:, :, :3]
    roi[:] = (alpha * rgb + (1 - alpha) * roi).astype(np.uint8)

def spawn_items(w, h, bins_y_top) -> List[ActiveItem]:
    items_pool: List[Item] = ITEMS.copy()
    random.shuffle(items_pool)
    items_pool = items_pool[:NUM_ITEMS_ON_SCREEN]

    active_items: List[ActiveItem] = []
    for it in items_pool:
        x = random.randint(ITEM_RADIUS + 40, w - ITEM_RADIUS - 40)
        y = random.randint(TOP_BANNER + ITEM_RADIUS + 40, bins_y_top - ITEM_RADIUS - 40)
        active_items.append(ActiveItem(item=it, pos=(x, y)))
    return active_items

def main():
    cap = cv.VideoCapture(0, cv.CAP_AVFOUNDATION)
    if not cap.isOpened():
        raise SystemExit("Could not open camera. Grant permissions.")

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cv.namedWindow(WINDOW_NAME, cv.WINDOW_NORMAL)

    active_items: List[ActiveItem] = []
    grabbed_index: Optional[int] = None
    hold_offset = (0, 0)
    score_correct = 0
    score_total = 0  

    msg = ""
    msg_until = 0.0
    msg_color = ACCENT

    game_start_time = time.time()
    remaining_time = GAME_TIME_LIMIT

    while True:
        ok, frame = cap.read()
        if not ok:
            cv.waitKey(1)
            continue

        frame = cv.flip(frame, 1)
        h, w = frame.shape[:2]

        rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        res = hands.process(rgb)

        tint = np.full_like(frame, BG, dtype=np.uint8)
        frame = cv.addWeighted(frame, 0.80, tint, 0.20, 0.0)

        cv.rectangle(frame, (0, 0), (w, TOP_BANNER), (45, 45, 60), -1)
        cv.putText(frame,
            "Multi-item sorting: pinch to grab an item and drop it into the correct bin",
            (12, 50),
            FONT,
            0.6,
            FG,
            2,
            cv.LINE_AA,
        )

        bins = []
        usable_w = w - (len(BINS) + 1) * BIN_GAP
        bin_w = max(80, usable_w // len(BINS))
        y1 = h - BIN_HEIGHT - 12
        for i, (label, color) in enumerate(BINS):
            x1 = BIN_GAP + i * (bin_w + BIN_GAP)
            x2 = x1 + bin_w
            y2 = y1 + BIN_HEIGHT
            bins.append((label, color, (x1, y1, x2, y2)))

        for label, color, (bx1, by1, bx2, by2) in bins:
            cv.rectangle(frame, (bx1, by1), (bx2, by2), color, -1)
            cv.rectangle(frame, (bx1, by1), (bx2, by2), (255, 255, 255), 2)
            cv.putText(frame, label, (bx1 + 10, by2 - 12), FONT, 0.7, (255, 255, 255), 2, cv.LINE_AA)

        if not active_items:
            active_items = spawn_items(w, h, y1)

        finger_tip_px = None
        if res.multi_hand_landmarks:
            lm = res.multi_hand_landmarks[0].landmark
            lm_px = {i: norm_to_px(lm[i], w, h) for i in [THUMB_TIP, INDEX_TIP, INDEX_MCP, PINKY_MCP]}
            is_pinched = pinch_state(lm_px, (0.3, 0.5), grabbed_index is not None)
            finger_tip_px = lm_px[INDEX_TIP]

            mp_draw.draw_landmarks(
                frame,
                res.multi_hand_landmarks[0],
                mp_hands.HAND_CONNECTIONS,
                mp_styles.get_default_hand_landmarks_style(),
                mp_styles.get_default_hand_connections_style(),
            )

            if grabbed_index is None and is_pinched and finger_tip_px is not None:
                closest_idx = None
                closest_dist = float("inf")
                for idx, a in enumerate(active_items):
                    if not a.active:
                        continue
                    d = l2(finger_tip_px, a.pos)
                    if d < closest_dist:
                        closest_dist = d
                        closest_idx = idx
                if closest_idx is not None and closest_dist <= (GRAB_RADIUS + ITEM_RADIUS):
                    grabbed_index = closest_idx
                    ax, ay = active_items[grabbed_index].pos
                    hold_offset = (ax - finger_tip_px[0], ay - finger_tip_px[1])

            if grabbed_index is not None and finger_tip_px is not None:
                new_x = finger_tip_px[0] + hold_offset[0]
                new_y = finger_tip_px[1] + hold_offset[1]
                active_items[grabbed_index].pos = (new_x, new_y)

                if not is_pinched:
                    dropped_item = active_items[grabbed_index]
                    dropped_label = bin_hit(dropped_item.pos, bins)
                    if dropped_label is not None:
                        score_total += 1
                        correct_label = dropped_item.item.correct_bin
                        if dropped_label == correct_label:
                            score_correct += 1
                            msg = f"Correct! {dropped_item.item.name} -> {correct_label}"
                            msg_color = (80, 200, 80)      # green
                            msg_until = time.time() + 1.2
                        else:
                            msg = f"Oops! {dropped_item.item.name} is {correct_label}"
                            msg_color = (60, 60, 255)      # red
                            msg_until = time.time() + 1.6
                        dropped_item.active = False
                    grabbed_index = None
                    hold_offset = (0, 0)

        for idx, a in enumerate(active_items):
            if not a.active:
                continue
            x = int(np.clip(a.pos[0], ITEM_RADIUS, w - ITEM_RADIUS))
            y = int(np.clip(a.pos[1], TOP_BANNER + ITEM_RADIUS, y1 - ITEM_RADIUS))
            a.pos = (x, y)

            img = ITEM_IMAGES.get(a.item.name)
            if img is not None:
                overlay_at_center(frame, img, a.pos)
            draw_text_centered(frame, a.item.name, (x, y - ITEM_RADIUS - 25), 0.8, (240, 240, 240), 2)

        now = time.time()
        elapsed = now - game_start_time
        remaining_time = max(0.0, GAME_TIME_LIMIT - elapsed)

        cv.putText(frame, f"Correct: {score_correct}/10", (w - 320, 30), FONT, 0.7, (120, 255, 120), 2, cv.LINE_AA)
        cv.putText(frame, f"Time: {int(remaining_time)}s", (w - 320, 55), FONT, 0.7, (255, 255, 255), 2, cv.LINE_AA)

        if time.time() < msg_until:
            draw_banner(frame, msg, msg_color)

        all_sorted = all(not a.active for a in active_items)

        if remaining_time <= 0.0 or all_sorted:
            final_screen(frame, score_correct)
            cv.imshow(WINDOW_NAME, frame)
            if cv.waitKey(1) & 0xFF == ord("q"):
                break
            continue

        cv.imshow(WINDOW_NAME, frame)
        if cv.waitKey(1) & 0xFF == ord("q"):
            break

    hands.close()
    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()