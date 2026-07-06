"""Render the suggested beta as an animated stick-figure GIF over the wall photo.

The figure is driven by the body-state timeline (LH/RH/LF/RF -> hold) that the
sequence generator tracks. Between two states, only the limbs that changed
animate — travelling in a slight arc — while the torso is re-derived every
frame from the four contact points, so the body follows naturally.
"""

import io
import math
from PIL import Image, ImageDraw, ImageFont

ACCENT  = "#fb923c"
OUTLINE = "#0f1011"
JOINTS  = ("LH", "RH", "LF", "RF")

FRAMES_PER_MOVE = 7   # animation frames per move
HOLD_FRAMES     = 4   # frames the pose is held after each move
INTRO_FRAMES    = 6   # frames showing the start position


def _lerp(a, b, t):
    return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)


def _ease(t):
    return t * t * (3 - 2 * t)  # smoothstep


def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _mid(pts):
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return (sum(xs) / len(xs), sum(ys) / len(ys))


def compute_keypose(state, hold_px, limb_len):
    """Concrete pixel positions for all four joints in a given body state.
    Feet that are flagging/smearing (hold None) hang relative to the body."""
    pts = {}
    for j in JOINTS:
        h = state.get(j)
        if h is not None and h in hold_px:
            pts[j] = hold_px[h]

    hands = [pts[j] for j in ("LH", "RH") if j in pts]
    if hands:
        hands_mid = _mid(hands)
    else:  # degenerate, should not happen — park mid-image
        hands_mid = (limb_len * 4, limb_len * 4)

    feet_known = [pts[j] for j in ("LF", "RF") if j in pts]
    if feet_known:
        feet_mid = _mid(feet_known)
    else:
        feet_mid = (hands_mid[0], hands_mid[1] + 2.2 * limb_len)

    hip = _lerp(hands_mid, feet_mid, 0.62)

    for j, side in (("LF", -1), ("RF", 1)):
        if j not in pts:
            pts[j] = (hip[0] + side * 0.9 * limb_len, hip[1] + 0.9 * limb_len)

    return pts


def _bend_point(root, tip, torso_x, amount=0.22):
    """Elbow/knee: midpoint pushed perpendicular, away from the torso axis."""
    mx, my = _mid([root, tip])
    dx, dy = tip[0] - root[0], tip[1] - root[1]
    length = math.hypot(dx, dy) or 1.0
    # both perpendicular candidates; pick the one pointing away from torso
    p1 = (-dy / length, dx / length)
    p2 = (dy / length, -dx / length)
    perp = p1 if (mx + p1[0] - torso_x) * (mx - torso_x + 0.001) >= 0 else p2
    off = amount * length
    return (mx + perp[0] * off, my + perp[1] * off)


def _draw_line(draw, a, b, width):
    draw.line([a, b], fill=OUTLINE, width=width + 3)
    draw.line([a, b], fill=ACCENT, width=width)


def _draw_limb(draw, root, tip, torso_x, width):
    bend = _bend_point(root, tip, torso_x)
    _draw_line(draw, root, bend, width)
    _draw_line(draw, bend, tip, width)


def draw_figure(draw, pts, limb_len, line_w):
    hands_mid = _mid([pts["LH"], pts["RH"]])
    feet_mid  = _mid([pts["LF"], pts["RF"]])

    shoulder = _lerp(hands_mid, feet_mid, 0.30)
    hip      = _lerp(hands_mid, feet_mid, 0.62)

    # head sits above the shoulders along the feet->hands axis
    ax, ay = hands_mid[0] - feet_mid[0], hands_mid[1] - feet_mid[1]
    alen = math.hypot(ax, ay) or 1.0
    head_c = (shoulder[0] + ax / alen * 0.30 * limb_len,
              shoulder[1] + ay / alen * 0.30 * limb_len)
    head_r = 0.20 * limb_len

    # torso
    _draw_line(draw, shoulder, hip, line_w + 1)

    # limbs
    _draw_limb(draw, shoulder, pts["LH"], shoulder[0], line_w)
    _draw_limb(draw, shoulder, pts["RH"], shoulder[0], line_w)
    _draw_limb(draw, hip,      pts["LF"], hip[0],      line_w)
    _draw_limb(draw, hip,      pts["RF"], hip[0],      line_w)

    # head
    draw.ellipse([head_c[0] - head_r - 2, head_c[1] - head_r - 2,
                  head_c[0] + head_r + 2, head_c[1] + head_r + 2], fill=OUTLINE)
    draw.ellipse([head_c[0] - head_r, head_c[1] - head_r,
                  head_c[0] + head_r, head_c[1] + head_r], fill=ACCENT)

    # hands/feet dots
    r = line_w + 2
    for j in JOINTS:
        x, y = pts[j]
        draw.ellipse([x - r, y - r, x + r, y + r], fill=OUTLINE)
        draw.ellipse([x - r + 1, y - r + 1, x + r - 1, y + r - 1], fill="#ffffff")


def _label_for(move):
    if move["move_number"] == 0:
        return "Start position"
    hold_str = f"hold {move['hold']}" if move["hold"] is not None else "wall"
    return f"Move {move['move_number']}: {move['limb']} → {hold_str}"


def _draw_label(draw, text, font, img_w):
    pad = 6
    try:
        box = draw.textbbox((0, 0), text, font=font)
        tw, th = box[2] - box[0], box[3] - box[1]
    except Exception:
        tw, th = len(text) * 7, 12
    draw.rectangle([10, 10, 10 + tw + pad * 2, 10 + th + pad * 2], fill=OUTLINE)
    draw.text((10 + pad, 10 + pad), text, fill="#ececec", font=font)


def render_beta_gif(base_image, holds, sequence, states, out_width=480, frame_ms=100):
    """Build the animated GIF. Returns io.BytesIO with the GIF bytes."""
    img = base_image.convert("RGB")
    scale = out_width / img.width
    base = img.resize((out_width, max(1, int(img.height * scale))))

    hold_px = {h["number"]: (h["x"] * scale, h["y"] * scale) for h in holds}
    limb_len = base.height * 0.075
    line_w = max(3, out_width // 160)

    try:
        font = ImageFont.load_default(size=max(12, out_width // 40))
    except TypeError:  # older Pillow
        font = ImageFont.load_default()

    keyposes = [compute_keypose(s, hold_px, limb_len) for s in states]

    # Shared global palette — per-frame palettes are what balloon GIF size.
    # Seed it with the figure colours so they survive quantisation.
    pal_src = base.copy()
    seed = ImageDraw.Draw(pal_src)
    # large swatches so median-cut gives these colours their own buckets
    sw = max(40, out_width // 8)
    for k, col in enumerate((ACCENT, OUTLINE, "#ffffff", "#ececec")):
        seed.rectangle([k * sw, 0, (k + 1) * sw - 1, sw - 1], fill=col)
    palette_img = pal_src.quantize(colors=128)

    frames = []

    def add_frame(pts, label, ring=None, ring_t=0.0):
        frame = base.copy()
        draw = ImageDraw.Draw(frame)
        if ring is not None:
            rad = 14 + 8 * math.sin(ring_t * math.pi)
            x, y = ring
            draw.ellipse([x - rad, y - rad, x + rad, y + rad],
                         outline=ACCENT, width=3)
        draw_figure(draw, pts, limb_len, line_w)
        _draw_label(draw, label, font, out_width)
        frames.append(frame.quantize(palette=palette_img, dither=Image.Dither.NONE))

    # intro: start position
    for _ in range(INTRO_FRAMES):
        add_frame(keyposes[0], _label_for(sequence[0]))

    # animate each move
    n = min(len(keyposes), len(sequence))
    for i in range(1, n):
        a, b = keyposes[i - 1], keyposes[i]
        changed = [j for j in JOINTS if _dist(a[j], b[j]) > 2]
        label = _label_for(sequence[i])

        # ring on the destination hold of the primary moving limb
        ring = None
        if sequence[i]["hold"] is not None and sequence[i]["hold"] in hold_px:
            ring = hold_px[sequence[i]["hold"]]

        for f in range(FRAMES_PER_MOVE):
            t = _ease((f + 1) / FRAMES_PER_MOVE)
            pts = {}
            for j in JOINTS:
                if j in changed:
                    pos = _lerp(a[j], b[j], t)
                    # arc lift, perpendicular to travel, biased upward
                    dx, dy = b[j][0] - a[j][0], b[j][1] - a[j][1]
                    seg = math.hypot(dx, dy) or 1.0
                    perp = (-dy / seg, dx / seg)
                    if perp[1] > 0:
                        perp = (-perp[0], -perp[1])
                    lift = 0.18 * seg * math.sin(t * math.pi)
                    pts[j] = (pos[0] + perp[0] * lift, pos[1] + perp[1] * lift)
                else:
                    pts[j] = a[j]
            add_frame(pts, label, ring, t)

        for _ in range(HOLD_FRAMES):
            add_frame(b, label)

    # linger on the final pose
    for _ in range(HOLD_FRAMES * 2):
        add_frame(keyposes[n - 1], "Send! \U0001F389")

    buf = io.BytesIO()
    frames[0].save(
        buf, format="GIF", save_all=True, append_images=frames[1:],
        duration=frame_ms, loop=0, optimize=True,
    )
    buf.seek(0)
    return buf
