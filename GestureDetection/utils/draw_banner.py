import cv2 as cv

def draw_banner(frame, text, color, font, height=80, pad=16, scale=1.0, thickness=2):
    h, w = frame.shape[:2]
    cv.rectangle(frame, (0, 0), (w, height), color, -1)
    cv.putText(frame, text, (pad, int(height * 0.65)), font, scale, (255, 255, 255), thickness, cv.LINE_AA)