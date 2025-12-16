import cv2 as cv

def draw_text_centered(frame, text, y, font, scale, color, thickness):
    h, w = frame.shape[:2]
    (tw, th), _ = cv.getTextSize(text, font, scale, thickness)
    x = (w - tw) // 2
    cv.putText(frame, text, (x, y), font, scale, color, thickness, cv.LINE_AA)