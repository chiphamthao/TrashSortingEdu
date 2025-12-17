import cv2 as cv

def norm_to_px(lm, w, h):
    return (int(lm.x * w), int(lm.y * h))