import cv2 as cv
def put_centered(img, text, center_bottom, scale, color, thickness, FONT):
    (tw, th), _ = cv.getTextSize(text, FONT, scale, thickness)
    x = int(center_bottom[0] - tw / 2)
    y = int(center_bottom[1])
    cv.putText(img, text, (x, y), FONT, scale, color, thickness, cv.LINE_AA)