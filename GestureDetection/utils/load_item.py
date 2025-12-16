import cv2 as cv
def load_item(path: str, r: int):
    img = cv.imread(path, cv.IMREAD_UNCHANGED)
    return cv.resize(img, (r * 4, r * 4))