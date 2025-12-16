"""
Part 1: Raise fingers (1 -> 4) to answer
"""
import cv2 as cv
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple
import mediapipe as mp
from utils.draw_banner import draw_banner
from utils.draw_text_centered import draw_text_centered

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

@dataclass
class Question:
    prompt: str
    choices: List[str]
    correct: str      

QUESTIONS: List[Question] = [
    Question(
        prompt="A greasy pizza box belongs in...",
        choices=["Recycling", "Compost", "Landfill", "E-waste"],
        correct="B",
    ),
    Question(
        prompt="An empty plastic bottle goes to...",
        choices=["Recycling", "Compost", "Landfill", "Special drop-off"],
        correct="A",
    ),
    Question(
        prompt="Old phone charger should be...",
        choices=["Recycling bin", "Landfill", "E-waste", "Compost"],
        correct="C",
    ),
]

WINDOW_NAME = "Waste Sorting Quiz"
FONT = cv.FONT_HERSHEY_SIMPLEX
COLOR_BG = (30, 30, 30)
COLOR_TEXT = (0.0, 0.0, 0.0)
COLOR_HINT = (180, 180, 180)
COLOR_OK = (60, 180, 75)
COLOR_BAD = (50, 50, 230)
COLOR_ACCENT = (128, 64, 255)

CONFIRMATION_SEC = 2.0   # how long the same finger count must be held to confirm
COOLDOWN_SEC = 1.0       # wait before accepting next selection after confirm

TIP_IDS = [8, 12, 16, 20]     # index, middle, ring, pinky tips
PIP_IDS = [6, 10, 14, 18]     # one joint below tip

def count_extended_fingers(landmarks):
    """
    A finger is up if tip.y < pip.y. Ignore thumb
    """
    up = 0
    for tip_id, pip_id in zip(TIP_IDS, PIP_IDS):
        tip = landmarks[tip_id]
        pip = landmarks[pip_id]
        if tip.y < pip.y:  
            up += 1
    return up

def fingers_to_choice(n_up: int):
    return {1: "A", 2: "B", 3: "C", 4: "D"}.get(n_up)

def draw_panel(frame, q: Question, choice_hover: Optional[str], progress: float, score: Tuple[int,int]):
    """
    Draw the question, choices, progress bar, and score.
    """
    h, w = frame.shape[:2]
    pad = 16

    # top banner
    cv.rectangle(frame, (0, 0), (w, 120), (45, 45, 60), thickness=-1)
    cv.putText(frame, "Waste Sorting Quiz", (pad, 40), FONT, 1.0, COLOR_TEXT, 2, cv.LINE_AA)
    cv.putText(frame, "Hold up 1-4 fingers. 1=A, 2=B, 3=C, 4=D", (pad, 75), FONT, 0.8, COLOR_HINT, 1, cv.LINE_AA)

    # score
    correct, total = score
    cv.putText(frame, f"Score: {correct}/{total}", (w - 220, 40), FONT, 0.9, COLOR_OK, 2, cv.LINE_AA)

    y0 = 170
    scale = 0.8
    thickness = 2
    cv.putText(frame, q.prompt, (pad, y0), FONT, scale, COLOR_TEXT, thickness, cv.LINE_AA)

    (_, text_h), _ = cv.getTextSize(q.prompt, FONT, scale, thickness)
    choices_y = y0 + text_h + 30  

    # choice boxes
    box_w = w - 2*pad
    box_h = 60
    gap = 14

    labels = ["A", "B", "C", "D"]
    for i, label in enumerate(labels):
        y1 = choices_y + i*(box_h + gap)
        y2 = y1 + box_h
        x1, x2 = pad, pad + box_w

        color = (70, 70, 90)
        if choice_hover == label:
            color = (90, 90, 140)

        cv.rectangle(frame, (x1, y1), (x2, y2), color, thickness=-1)
        cv.rectangle(frame, (x1, y1), (x2, y2), (120, 120, 150), thickness=2)
        cv.putText(frame, f"{label}. {q.choices[i]}", (x1 + 12, y1 + 40),
                   FONT, 0.8, COLOR_TEXT, 2, cv.LINE_AA)

    # progress bar
    bar_w, bar_h = w - 2*pad, 14
    bx1, by1 = pad, h - pad - bar_h
    bx2 = int(bx1 + bar_w * max(0.0, min(1.0, progress)))
    cv.rectangle(frame, (bx1, by1), (bx1 + bar_w, by1 + bar_h), (60, 60, 80), -1)
    if progress > 0:
        cv.rectangle(frame, (bx1, by1), (bx2, by1 + bar_h), COLOR_ACCENT, -1)

def run_quiz():
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        raise SystemExit("Could not open camera. Check permissions.")

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        model_complexity=1,
        min_detection_confidence=0.5 ,
        min_tracking_confidence=0.5 ,
    )

    cv.namedWindow(WINDOW_NAME, cv.WINDOW_NORMAL)

    q_idx = 0
    score_correct = 0
    total_answered = 0

    last_choice = None
    choice_start_t = None
    last_confirm_t = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv.flip(frame, 1) 
        rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        results = hands.process(rgb)

        n_up = 0
        hover_choice = None

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            n_up = count_extended_fingers(hand_landmarks.landmark)
            hover_choice = fingers_to_choice(n_up)

            # draw landmarks
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_styles.get_default_hand_landmarks_style(),
                mp_styles.get_default_hand_connections_style(),
            )

        # wait logic
        now = time.time()
        progress = 0.0
        if hover_choice is None or (now - last_confirm_t) < COOLDOWN_SEC:
            last_choice = None
            choice_start_t = None
        else:
            if hover_choice != last_choice:
                last_choice = hover_choice
                choice_start_t = now
            else:
                progress = min(1.0, (now - (choice_start_t or now)) / CONFIRMATION_SEC)
                if progress >= 1.0 and (now - last_confirm_t) >= COOLDOWN_SEC:
                    # Confirm selection
                    confirmed = hover_choice
                    last_confirm_t = now
                    # Check answer
                    q = QUESTIONS[q_idx]
                    is_correct = (confirmed == q.correct)
                    total_answered += 1
                    if is_correct:
                        score_correct += 1
                        draw_banner(frame, f"{confirmed} is correct!", COLOR_OK)
                    else:
                        draw_banner(frame, f"{confirmed} is wrong. Correct: {q.correct}", COLOR_BAD)
                    cv.imshow(WINDOW_NAME, frame)
                    cv.waitKey(500) 

                    # Next question
                    q_idx += 1
                    if q_idx >= len(QUESTIONS):
                        # Quiz finished
                        show_final(frame, score_correct, total_answered)
                        cv.imshow(WINDOW_NAME, frame)
                        key = cv.waitKey(2500) & 0xFF
                        break

                    # reset progress
                    last_choice = None
                    choice_start_t = None
                    progress = 0.0

        draw_panel(frame,QUESTIONS[q_idx],hover_choice,progress,(score_correct, total_answered),)
        cv.imshow(WINDOW_NAME, frame)
        key = cv.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    hands.close()
    cap.release()
    cv.destroyAllWindows()

def show_final(frame, correct: int, total: int):
    h, w = frame.shape[:2]
    cv.rectangle(frame, (0, 0), (w, h), COLOR_BG, thickness=-1)
    msg = "Quiz Complete!"
    score_msg = f"Score: {correct}/{total}"
    hint_msg = "Press 'q' to close"
    center_y = h // 2

    draw_text_centered(frame, msg, center_y - 40, FONT, 1.8, (255, 255, 255), 3)
    draw_text_centered(frame, score_msg, center_y + 10,  FONT, 1.3, COLOR_OK, 3)
    draw_text_centered(frame, hint_msg, center_y + 60,  FONT, 0.9, COLOR_HINT, 2)

if __name__ == "__main__":
    run_quiz()