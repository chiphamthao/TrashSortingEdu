import cv2 as cv

def compute_bins(w, h, bin_specs, bin_gap, bin_height, bottom_pad=12):
    bins = []
    usable_w = w - (len(bin_specs) + 1) * bin_gap
    bin_w = max(80, usable_w // len(bin_specs))
    y1 = h - bin_height - bottom_pad

    for i, (label, color) in enumerate(bin_specs):
        x1 = bin_gap + i * (bin_w + bin_gap)
        x2 = x1 + bin_w
        y2 = y1 + bin_height
        bins.append((label, color, (x1, y1, x2, y2)))

    return bins, y1


def draw_bins(frame, bins, font, label_scale=0.7, label_thickness=2):
    for label, color, (x1, y1, x2, y2) in bins:
        cv.rectangle(frame, (x1, y1), (x2, y2), color, -1)
        cv.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
        cv.putText(
            frame,
            label,
            (x1 + 10, y2 - 12),
            font,
            label_scale,
            (255, 255, 255),
            label_thickness,
            cv.LINE_AA,
        )