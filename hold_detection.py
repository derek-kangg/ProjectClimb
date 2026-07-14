"""OpenCV hold-candidate detection.

Finds colour-matched hold candidates with pixel-accurate coordinates.
Includes two shape-level corrections learned from real gym photos:
- watershed splitting for merged blobs (two touching holds detected as one)
- a tape filter (route tape is route-coloured but thin and elongated)
Claude's vision pass (in app.py) then labels type/use/orientation.
"""

import cv2
import numpy as np

COLOUR_RANGES = {
    "Black":  [(0, 0, 0),      (180, 80, 50)],
    "Blue":   [(90, 50, 50),   (130, 255, 255)],
    "Red":    [(0, 100, 100),  (10, 255, 255)],
    "Green":  [(40, 50, 50),   (80, 255, 255)],
    "Orange": [(10, 100, 100), (25, 255, 255)],
    "Pink":   [(140, 50, 100), (170, 255, 255)],
    "White":  [(0, 0, 180),    (180, 30, 255)],
    "Yellow": [(25, 100, 100), (35, 255, 255)],
    "Purple": [(130, 50, 50),  (160, 255, 255)],
}


def _is_tape(contour, img_width):
    """Route tape: thin, elongated strip of the route colour."""
    (_, _), (rw, rh), _ = cv2.minAreaRect(contour)
    short, long_ = min(rw, rh), max(rw, rh)
    if short <= 0:
        return True
    max_tape_width = max(16, img_width * 0.025)
    return (long_ / short) > 2.3 and short < max_tape_width


def _split_merged_blob(mask, contour):
    """Try to split one contour that may contain 2+ touching holds.

    Watershed on distance-transform cores, then PAIRWISE neck analysis:
    two adjacent pieces re-merge when the neck between them is nearly as
    thick as the smaller piece's core (one chalky/patchy hold), and stay
    apart when they meet at a thin pinch (two real holds). Junk cores get
    absorbed by their neighbours instead of poisoning a global test.

    Returns a list of contours (full-image coordinates) if a genuine split
    remains, else None."""
    x, y, w, h = cv2.boundingRect(contour)

    # Isolated ROI mask of just this contour (not neighbours)
    roi = np.zeros((h, w), dtype=np.uint8)
    shifted = contour - [x, y]
    cv2.drawContours(roi, [shifted], -1, 255, thickness=cv2.FILLED)

    dist = cv2.distanceTransform(roi, cv2.DIST_L2, 5)
    peak = dist.max()
    if peak <= 0:
        return None

    core_thresh = max(0.35 * peak, 5.0)
    _, sure_fg = cv2.threshold(dist, core_thresh, 255, 0)
    sure_fg = np.uint8(sure_fg)
    n_labels, markers = cv2.connectedComponents(sure_fg)
    if n_labels <= 2:  # background + one core -> genuinely one hold
        return None

    # Watershed to assign the ambiguous pixels between the cores
    markers = markers + 1
    unknown = cv2.subtract(roi, sure_fg)
    markers[unknown == 255] = 0
    cv2.watershed(cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR), markers)

    labels = [lb for lb in range(2, n_labels + 1) if np.any(markers == lb)]
    if len(labels) < 2:
        return None

    piece_masks = {lb: np.uint8(markers == lb) for lb in labels}
    peaks = {lb: float(dist[markers == lb].max()) for lb in labels}
    kern = np.ones((5, 5), np.uint8)
    dilated = {lb: cv2.dilate(piece_masks[lb], kern) for lb in labels}

    # union-find over pieces; merge pairs whose shared neck is fat
    parent = {lb: lb for lb in labels}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i, a in enumerate(labels):
        for b in labels[i + 1:]:
            shared = (dilated[a] > 0) & (dilated[b] > 0) & (roi > 0)
            if not shared.any():
                continue
            neck = float(dist[shared].max())
            if neck >= 0.65 * min(peaks[a], peaks[b]):
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[ra] = rb

    groups = {}
    for lb in labels:
        groups.setdefault(find(lb), []).append(lb)
    if len(groups) < 2:
        return None  # everything merged back -> one hold

    blob_area = cv2.contourArea(contour)
    pieces = []
    for members in groups.values():
        gmask = np.zeros_like(roi)
        for lb in members:
            gmask = cv2.bitwise_or(gmask, piece_masks[lb] * 255)
        cs, _ = cv2.findContours(gmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
                                 offset=(x, y))
        if not cs:
            continue
        biggest = max(cs, key=cv2.contourArea)
        # keep only substantial pieces — slivers mean a bad split
        if cv2.contourArea(biggest) > 0.025 * blob_area:
            pieces.append(biggest)

    return pieces if len(pieces) >= 2 else None


def find_hold_candidates(image_path, colour, min_area=80):
    """Detect hold candidates for a route colour. Returns a list of dicts:
    {bx, by, bw, bh, x, y, size} in full-image pixel coordinates."""
    cv_image = cv2.imread(image_path)
    img_h, img_w = cv_image.shape[:2]
    hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

    low, high = COLOUR_RANGES[colour]
    mask = cv2.inRange(hsv, np.array(low), np.array(high))

    # CLOSE kernel scales with resolution so chalk patches on a hold do not
    # fragment it into multiple detections (phone photos are ~2000px+)
    k = max(5, img_w // 200) | 1
    close_kernel = np.ones((k, k), np.uint8)
    open_kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Split any large blob that actually contains multiple touching holds.
    # Scale the "large" threshold with resolution (phone photos are ~2000px+).
    split_threshold = max(1500, (img_w / 700.0) ** 2 * 350)
    processed = []  # (contour, split_group) — pieces of one split share a group id
    split_group = 0
    for contour in contours:
        if cv2.contourArea(contour) >= split_threshold:
            pieces = _split_merged_blob(mask, contour)
            if pieces:
                split_group += 1
                processed.extend((p, split_group) for p in pieces)
                continue
        processed.append((contour, None))

    candidates = []
    for contour, grp in processed:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue

        bx, by, bw, bh = cv2.boundingRect(contour)
        if bw < 12 or bh < 12:
            continue
        if by < img_h * 0.03:
            continue
        if _is_tape(contour, img_w):
            continue

        cx, cy = bx + bw // 2, by + bh // 2

        # Size classes relative to resolution (original tuning was ~770px wide)
        scale2 = (img_w / 770.0) ** 2
        if area < 500 * scale2:
            size = "small"
        elif area < 2000 * scale2:
            size = "medium"
        else:
            size = "large"

        candidates.append({"bx": bx, "by": by, "bw": bw, "bh": bh,
                           "x": cx, "y": cy, "size": size, "grp": grp})

    deduped = _dedup_overlaps(candidates)
    for c in deduped:
        c.pop("grp", None)
    return deduped


_SIZE_RANK = {"small": 0, "medium": 1, "large": 2}


def _dedup_overlaps(candidates):
    """Merge candidates whose boxes substantially overlap (>=40% of the
    smaller box) — usually chalk-fragmented pieces of one hold."""
    merged = True
    while merged:
        merged = False
        for i in range(len(candidates)):
            for j in range(i + 1, len(candidates)):
                a, b = candidates[i], candidates[j]
                # never re-merge pieces the watershed deliberately split apart
                if a.get("grp") is not None and a.get("grp") == b.get("grp"):
                    continue
                ix = max(0, min(a["bx"] + a["bw"], b["bx"] + b["bw"]) - max(a["bx"], b["bx"]))
                iy = max(0, min(a["by"] + a["bh"], b["by"] + b["bh"]) - max(a["by"], b["by"]))
                inter = ix * iy
                smaller = min(a["bw"] * a["bh"], b["bw"] * b["bh"])
                if smaller > 0 and inter / smaller >= 0.4:
                    bx = min(a["bx"], b["bx"])
                    by = min(a["by"], b["by"])
                    bw = max(a["bx"] + a["bw"], b["bx"] + b["bw"]) - bx
                    bh = max(a["by"] + a["bh"], b["by"] + b["bh"]) - by
                    candidates[i] = {
                        "bx": bx, "by": by, "bw": bw, "bh": bh,
                        "x": bx + bw // 2, "y": by + bh // 2,
                        "size": max(a["size"], b["size"], key=lambda s: _SIZE_RANK[s]),
                    }
                    del candidates[j]
                    merged = True
                    break
            if merged:
                break
    return candidates
