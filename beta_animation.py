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
BODY    = "#ececec"
JOINTS  = ("LH", "RH", "LF", "RF")

LIMB_COLOURS = {
    "LH": "#facc15",  # left hand  — yellow
    "RH": "#fb923c",  # right hand — orange
    "LF": "#4ade80",  # left foot  — green
    "RF": "#38bdf8",  # right foot — blue
}
LIMB_NAMES = {"LH": "L hand", "RH": "R hand", "LF": "L foot", "RF": "R foot"}


def _limb_joint(limb):
    """Map a sequence limb string to its joint key (None for both/unknown)."""
    if "left" in limb:
        return "LH" if "hand" in limb else "LF"
    if "right" in limb:
        return "RH" if "hand" in limb else "RF"
    return None

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


def _draw_line(draw, a, b, width, colour):
    draw.line([a, b], fill=OUTLINE, width=width + 3)
    draw.line([a, b], fill=colour, width=width)


def _draw_limb(draw, root, tip, torso_x, width, colour):
    bend = _bend_point(root, tip, torso_x)
    _draw_line(draw, root, bend, width, colour)
    _draw_line(draw, bend, tip, width, colour)


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

    # torso — neutral so the coloured limbs pop
    _draw_line(draw, shoulder, hip, line_w + 1, BODY)

    # limbs, colour-coded
    _draw_limb(draw, shoulder, pts["LH"], shoulder[0], line_w, LIMB_COLOURS["LH"])
    _draw_limb(draw, shoulder, pts["RH"], shoulder[0], line_w, LIMB_COLOURS["RH"])
    _draw_limb(draw, hip,      pts["LF"], hip[0],      line_w, LIMB_COLOURS["LF"])
    _draw_limb(draw, hip,      pts["RF"], hip[0],      line_w, LIMB_COLOURS["RF"])

    # head
    draw.ellipse([head_c[0] - head_r - 2, head_c[1] - head_r - 2,
                  head_c[0] + head_r + 2, head_c[1] + head_r + 2], fill=OUTLINE)
    draw.ellipse([head_c[0] - head_r, head_c[1] - head_r,
                  head_c[0] + head_r, head_c[1] + head_r], fill=BODY)

    # hands/feet dots in their limb colour
    r = line_w + 3
    for j in JOINTS:
        x, y = pts[j]
        draw.ellipse([x - r, y - r, x + r, y + r], fill=OUTLINE)
        draw.ellipse([x - r + 1, y - r + 1, x + r - 1, y + r - 1],
                     fill=LIMB_COLOURS[j])


def _draw_legend(draw, font, img_w):
    """Small colour legend, top-right corner."""
    pad, row_h, sw = 5, 15, 9
    entries = [(LIMB_COLOURS[j], LIMB_NAMES[j]) for j in JOINTS]
    w = 74
    h = pad * 2 + row_h * len(entries)
    x0 = img_w - w - 8
    y0 = 8
    draw.rectangle([x0, y0, x0 + w, y0 + h], fill=OUTLINE)
    for i, (col, name) in enumerate(entries):
        y = y0 + pad + i * row_h
        draw.ellipse([x0 + pad, y + 2, x0 + pad + sw, y + 2 + sw], fill=col)
        draw.text((x0 + pad + sw + 5, y), name, fill="#ececec", font=font)


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
        legend_font = ImageFont.load_default(size=11)
    except TypeError:  # older Pillow
        font = ImageFont.load_default()
        legend_font = font

    keyposes = [compute_keypose(s, hold_px, limb_len) for s in states]

    # Shared global palette — per-frame palettes are what balloon GIF size.
    # Seed it with the figure colours so they survive quantisation.
    pal_src = base.copy()
    seed = ImageDraw.Draw(pal_src)
    # large swatches so median-cut gives these colours their own buckets
    seed_colours = (ACCENT, OUTLINE, "#ffffff", BODY) + tuple(LIMB_COLOURS.values())
    sw = max(30, out_width // 10)
    for k, col in enumerate(seed_colours):
        seed.rectangle([k * sw, 0, (k + 1) * sw - 1, sw - 1], fill=col)
    palette_img = pal_src.quantize(colors=128)

    frames = []

    def add_frame(pts, label, ring=None, ring_t=0.0, ring_colour=ACCENT):
        frame = base.copy()
        draw = ImageDraw.Draw(frame)
        if ring is not None:
            rad = 14 + 8 * math.sin(ring_t * math.pi)
            x, y = ring
            draw.ellipse([x - rad, y - rad, x + rad, y + rad],
                         outline=ring_colour, width=3)
        draw_figure(draw, pts, limb_len, line_w)
        _draw_label(draw, label, font, out_width)
        _draw_legend(draw, legend_font, out_width)
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

        # ring on the destination hold, coloured by the moving limb
        ring = None
        joint = _limb_joint(sequence[i]["limb"])
        ring_colour = LIMB_COLOURS.get(joint, ACCENT)
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
            add_frame(pts, label, ring, t, ring_colour)

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
