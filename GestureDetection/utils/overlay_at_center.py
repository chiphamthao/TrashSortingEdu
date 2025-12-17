import numpy as np

def overlay_at_center(frame, overlay, center, TOP_BANNER):
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