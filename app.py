from route_graph import build_reachability_graph, format_graph_for_prompt
from beta_animation import render_beta_gif
from PIL import Image, ImageDraw
import streamlit as st
import anthropic
from dotenv import load_dotenv
import cv2
import numpy as np
import os
import base64
import tempfile
import io

load_dotenv()

def get_api_key():
    key = os.getenv("ANTHROPIC_API_KEY")
    if not key:
        try:
            key = st.secrets["ANTHROPIC_API_KEY"]
        except (KeyError, FileNotFoundError):
            key = None
    return key

anthropic_client = anthropic.Anthropic(api_key=get_api_key())

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


def detect_and_validate_holds(image_path, colour):
    """
    Two-pass hold detection:
    Pass 1 — OpenCV finds candidates by colour (pixel-accurate coordinates).
    Pass 2 — Claude Sonnet sees the annotated image, removes false positives,
              and identifies hold types. All in one vision call.
    Returns: (annotated PIL image, validated holds list with hold_type / best_use)
    """
    # --- Pass 1: OpenCV colour detection ---
    cv_image = cv2.imread(image_path)
    hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
    low, high = COLOUR_RANGES[colour]
    mask = cv2.inRange(hsv, np.array(low), np.array(high))
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 80:
            continue
        bx, by, bw, bh = cv2.boundingRect(contour)
        if bw < 12 or bh < 12:
            continue
        if by < cv_image.shape[0] * 0.03:
            continue
        cx, cy = bx + bw // 2, by + bh // 2
        size = "small" if area < 500 else "medium" if area < 2000 else "large"
        candidates.append({"bx": bx, "by": by, "bw": bw, "bh": bh, "x": cx, "y": cy, "size": size})

    if not candidates:
        return Image.open(image_path).convert("RGB"), []

    # Draw numbered candidates on a preview image for Claude to assess
    preview = Image.open(image_path).convert("RGB")
    preview_draw = ImageDraw.Draw(preview)
    for i, c in enumerate(candidates, 1):
        cx, cy = c["x"], c["y"]
        preview_draw.ellipse([cx-15, cy-15, cx+15, cy+15], fill="#fb923c")
        preview_draw.text((cx-5, cy-8), str(i), fill="black")

    buf = io.BytesIO()
    preview.save(buf, format="JPEG")
    image_data = base64.b64encode(buf.getvalue()).decode("utf-8")

    candidate_list = "\n".join(
        f"Candidate {i}: x={c['x']}, y={c['y']}, size={c['size']}"
        for i, c in enumerate(candidates, 1)
    )

    # --- Pass 2: Claude validation ---
    tool = {
        "name": "submit_validated_holds",
        "description": "Submit validation results for each detected hold candidate",
        "input_schema": {
            "type": "object",
            "properties": {
                "holds": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "number":    {"type": "integer"},
                            "hold_type": {"type": "string", "enum": ["jug", "crimp", "sloper", "pinch", "pocket", "edge", "volume", "chip", "unknown"]},
                            "best_use":  {"type": "string", "enum": ["handhold", "foothold", "both"]}
                        },
                        "required": ["number", "hold_type", "best_use"]
                    }
                }
            },
            "required": ["holds"]
        }
    }

    response = anthropic_client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        tools=[tool],
        tool_choice={"type": "any"},
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": image_data}},
                {"type": "text", "text": f"""This climbing wall image has {len(candidates)} numbered candidate holds detected by colour analysis (target colour: {colour}).

{candidate_list}

For every candidate number, identify:
1. Hold type: jug / crimp / sloper / pinch / pocket / edge / volume / chip
2. Best use: handhold, foothold, or both (small chips are typically footholds)

Assess every single candidate — do not skip any."""}
            ]
        }]
    )

    validated = {}
    for block in response.content:
        if block.type == "tool_use" and block.name == "submit_validated_holds":
            for h in block.input["holds"]:
                validated[h["number"]] = h
            break

    # Build final holds list and annotated image
    final_image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(final_image)
    final_holds = []
    new_number = 1

    for i, c in enumerate(candidates, 1):
        v = validated.get(i, {"hold_type": "unknown", "best_use": "both"})

        cx, cy = c["x"], c["y"]
        bx, by, bw, bh = c["bx"], c["by"], c["bw"], c["bh"]
        best_use  = v.get("best_use", "both")
        hold_type = v.get("hold_type", "unknown")

        if best_use == "foothold" or c["size"] == "small":
            size_label = "small (likely foothold)"
        elif c["size"] == "large":
            size_label = "large (likely handhold)"
        else:
            size_label = "medium (likely handhold)"

        final_holds.append({
            "number":    new_number,
            "x":         cx,
            "y":         cy,
            "size":      size_label,
            "hold_type": hold_type,
            "best_use":  best_use,
        })

        draw.rectangle([bx, by, bx + bw, by + bh], outline="#fb923c", width=3)
        draw.ellipse([cx-15, cy-15, cx+15, cy+15], fill="#fb923c")
        draw.text((cx-5, cy-8), str(new_number), fill="black")
        new_number += 1

    return final_image, final_holds


def build_holds_description(holds):
    if not holds:
        return "No holds detected."
    sorted_holds = sorted(holds, key=lambda h: h["y"], reverse=True)
    all_x = [h["x"] for h in holds]
    mid_x = (max(all_x) + min(all_x)) / 2

    lines = ["Holds listed bottom to top:"]
    for h in sorted_holds:
        side      = "LEFT" if h["x"] < mid_x else "RIGHT"
        hold_type = h.get("hold_type", "unknown")
        best_use  = h.get("best_use", "both")
        size      = h.get("size", "unknown")
        lines.append(
            f"Hold {h['number']}: {hold_type}, {size}, {side} side, best_use={best_use}, x={h['x']}, y={h['y']}"
        )
    return "\n".join(lines)


def apply_limb_to_state(new_state, limb, hold):
    """Apply a single limb placement to state dict in place."""
    if limb == "right hand":
        new_state["RH"] = hold
    elif limb == "left hand":
        new_state["LH"] = hold
    elif limb == "both hands":
        new_state["RH"] = hold
        new_state["LH"] = hold
    elif limb == "right foot":
        new_state["RF"] = hold
        if hold is not None and new_state["LF"] == hold:
            new_state["LF"] = None
    elif limb == "left foot":
        new_state["LF"] = hold
        if hold is not None and new_state["RF"] == hold:
            new_state["RF"] = None
    elif limb == "both feet":
        new_state["RF"] = hold
        new_state["LF"] = hold
    elif limb == "swap right foot":
        new_state["RF"] = hold
        new_state["LF"] = None
    elif limb == "swap left foot":
        new_state["LF"] = hold
        new_state["RF"] = None
    elif limb in ("smear right foot", "flag right foot"):
        new_state["RF"] = None
    elif limb in ("smear left foot", "flag left foot"):
        new_state["LF"] = None


def apply_move_to_state(state, move):
    """Apply a move to the body state dict."""
    new_state = state.copy()
    apply_limb_to_state(new_state, move["limb"], move["hold"])
    return new_state


def format_state(state):
    def fmt(v):
        return f"Hold {v}" if v is not None else "wall/air"
    return f"LH: {fmt(state['LH'])} | RH: {fmt(state['RH'])} | LF: {fmt(state['LF'])} | RF: {fmt(state['RF'])}"


def generate_sequence_iteratively(
    hold_descriptions, holds, graph, start_holds,
    height_cm, difficulty, wall_angle, finish_style, extra_notes,
    image_height=None, progress_callback=None
):
    """
    Generate a route sequence one move at a time.
    Python tracks body state; the model only picks the next single move.
    """
    MAX_MOVES = 20
    hold_map = {h["number"]: h for h in holds}
    finish_hold_num = min(holds, key=lambda h: h["y"])["number"]

    # Full body extension (feet to hand) in pixels — same 4m wall-scale
    # assumption route_graph uses. Guards against endless hand moves with
    # planted feet, which no human can do.
    if image_height:
        pixels_per_cm = image_height / 400.0
        max_span_px = height_cm * 1.30 * pixels_per_cm
    else:
        max_span_px = None

    # Auto-place starting feet on the lowest available holds near the start holds.
    # Prefer holds below (higher y) and near the x position of the start hands.
    start_x = sum(h["x"] for h in start_holds) / len(start_holds)
    start_y = max(h["y"] for h in start_holds)  # y of lowest start hand hold
    start_hand_nums = {h["number"] for h in start_holds}

    foot_candidates = sorted(
        [h for h in holds if h["y"] >= start_y - 30 and h["number"] not in start_hand_nums],
        key=lambda h: (-h["y"], abs(h["x"] - start_x))
    )

    lf_hold = foot_candidates[0]["number"] if len(foot_candidates) > 0 else None
    rf_hold = foot_candidates[1]["number"] if len(foot_candidates) > 1 else None

    if len(start_holds) == 1:
        state = {"LH": start_holds[0]["number"], "RH": start_holds[0]["number"], "LF": lf_hold, "RF": rf_hold}
        foot_cue = f"LF Hold {lf_hold}, RF {'Hold ' + str(rf_hold) if rf_hold else 'smear'}"
        start_cue = f"Both hands Hold {start_holds[0]['number']}, {foot_cue}"
        start_move = {"move_number": 0, "limb": "both hands", "hold": start_holds[0]["number"], "action": "start", "cue": start_cue}
    else:
        state = {"LH": start_holds[0]["number"], "RH": start_holds[1]["number"], "LF": lf_hold, "RF": rf_hold}
        foot_cue = f"LF Hold {lf_hold}, RF {'Hold ' + str(rf_hold) if rf_hold else 'smear'}"
        start_cue = f"LH Hold {start_holds[0]['number']}, RH Hold {start_holds[1]['number']}, {foot_cue}"
        start_move = {"move_number": 0, "limb": "both hands", "hold": None, "action": "start", "cue": start_cue}

    sequence = [start_move]
    states = [dict(state)]  # body-state timeline, one entry per sequence step

    angle_context = {
        "Slab (less than vertical)": "slab — trust feet, stand tall, smearing common",
        "Vertical": "vertical — balance arm and leg use equally",
        "Slight overhang": "slight overhang — hips close to wall, feet still important",
        "Steep overhang": "steep overhang — body tension critical, move efficiently",
        "Roof": "roof — heel hooks, toe hooks, core tension essential",
    }
    finish_context = {
        "TOP label":                    "marked TOP",
        "Tape on hold":                 "marked with tape",
        "Top-out (both hands on top)":  "top-out, both hands on wall top",
        "Not marked":                   "highest hold of this colour",
    }
    extra = f"\nExtra notes: {extra_notes}" if extra_notes.strip() else ""

    hold_positions = "\n".join(
        f"Hold {h['number']}: x={h['x']}, y={h['y']}, size={h['size']}, type={h.get('hold_type','?')}, use={h.get('best_use','both')}"
        for h in sorted(holds, key=lambda x: x["y"], reverse=True)
    )

    move_tool = {
        "name": "submit_move",
        "description": "Submit the single best next climbing move given the current body state. A move can include a simultaneous foot adjustment when the hand move naturally requires it.",
        "input_schema": {
            "type": "object",
            "properties": {
                "limb": {
                    "type": "string",
                    "enum": [
                        "right hand", "left hand", "both hands",
                        "right foot", "left foot", "both feet",
                        "flag right foot", "flag left foot",
                        "smear right foot", "smear left foot",
                        "swap right foot", "swap left foot"
                    ]
                },
                "hold": {
                    "type": ["integer", "null"],
                    "description": "Hold number, or null for smear/flag against wall"
                },
                "action": {
                    "type": "string",
                    "enum": ["move", "match", "flag", "smear", "swap", "heel hook", "toe hook", "drop knee", "deadpoint", "dynamic", "finish"]
                },
                "cue": {
                    "type": "string",
                    "description": "One short technique cue under 10 words"
                }
            },
            "required": ["limb", "hold", "action", "cue"]
        }
    }

    for move_num in range(1, MAX_MOVES + 1):
        if progress_callback:
            progress_callback(move_num, MAX_MOVES)

        def reachable_str(hold_num):
            if hold_num is None:
                return "none"
            entries = graph.get(hold_num, [])
            if not entries:
                return "none in reach"
            return ", ".join(
                f"Hold {r['hold']} ({r['difficulty']}, {r['direction']})"
                for r in entries[:8]
            )

        history = "\n".join(
            f"  {m['move_number']}: {m['limb']} → {'Hold ' + str(m['hold']) if m['hold'] is not None else 'wall'} ({m['action']}) — {m.get('cue', '')}"
            for m in sequence
        )

        if difficulty == "Beginner":
            technique_section = """TECHNIQUE LIBRARY (low-grade climb — keep it simple and controlled):
  Footwork: step onto hold | flag for balance | smear on slab | foot swap when needed
  Hands: match when repositioning is needed
  Movement: static moves only — no jumping or lunging
  Avoid heel hooks, toe hooks, drop knees, and dynamic moves unless the hold layout makes them unavoidable."""
        elif difficulty == "Intermediate":
            technique_section = """TECHNIQUE LIBRARY (intermediate grade — use technique when it clearly helps):
  Footwork: step onto hold | backstep (outside edge, hip in) | flag (inside/outside) | heel hook on obvious placements | smear | foot swap
  Hands: match | side pull | undercling
  Movement: deadpoint if a hold is just out of static reach
  Use advanced moves (drop knee, toe hook) only when the geometry clearly calls for it."""
        else:
            technique_section = """TECHNIQUE LIBRARY (advanced/high-grade climb — full repertoire expected):
  Footwork: step | heel hook (heel on/above hold, pull with hamstring) | toe hook (top of foot, pull) | drop knee (rotate knee in, hip drops, extends reach) | backstep (outside edge, hip turned in) | foot swap
  Balance: inside flag | outside flag | smear
  Movement: deadpoint (lunge, grip at apex) | dynamic (jump when hold is out of static reach)
  Hands: match | undercling (palm up, pull toward body) | side pull | gaston (elbow out, push away)
  Drop knees, heel hooks, and flags are expected — use them proactively for better position."""

        prompt = f"""You are generating a bouldering sequence one move at a time.

CURRENT BODY STATE:
{format_state(state)}

MOVE HISTORY:
{history}

FINISH: Hold {finish_hold_num} ({finish_context[finish_style]})
WALL: {angle_context[wall_angle]}
CLIMBER: {height_cm}cm, {difficulty}{extra}

HOLD INFO (type, size, position):
{hold_positions}

REACHABLE FROM CURRENT HANDS:
From LH (Hold {state['LH']}): {reachable_str(state['LH'])}
From RH (Hold {state['RH']}): {reachable_str(state['RH'])}

REACHABLE FROM CURRENT FEET (for foot moves):
From LF ({('Hold ' + str(state['LF'])) if state['LF'] else 'wall'}): {reachable_str(state['LF'])}
From RF ({('Hold ' + str(state['RF'])) if state['RF'] else 'wall'}): {reachable_str(state['RF'])}

MECHANICS:
- One limb moves per step — always.
- Two hands CAN share a hold (match)
- Two feet CANNOT share a small chip — foot swap instead
- Only move feet to holds in the foot reachable lists above, or smear/flag on wall
- As your hands climb, your feet MUST follow — a hand hold beyond full body extension from your feet is unreachable and will be rejected

FOOT ECONOMY — feet are support, hands drive progress:
- HANDS make upward progress. Move a hand on most moves.
- Only move a foot when the current foot position genuinely cannot support the next hand reach.
- Do NOT move feet proactively or to "prepare" — if feet are stable, leave them and move a hand instead.
- A pivot foothold (a good mid-height chip) should be reused via swaps as hands climb — do not abandon it for a new one each move.
- Flag the free foot when one foot is anchored on the pivot — flagging is stable, not a last resort.
- If in doubt between a foot move and a hand move: move the hand.

{technique_section}
Only move hands to holds shown in the reachable lists above.
When a hand reaches Hold {finish_hold_num}, output action=finish.

What is the single best next move?"""

        def request_move(extra_correction=""):
            messages = [{"role": "user", "content": prompt}]
            if extra_correction:
                messages.append({"role": "assistant", "content": [{"type": "text", "text": "I'll reconsider."}]})
                messages.append({"role": "user", "content": f"CORRECTION NEEDED: {extra_correction} Please submit a different move."})
            resp = anthropic_client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=256,
                tools=[move_tool],
                tool_choice={"type": "any"},
                messages=messages
            )
            for block in resp.content:
                if block.type == "tool_use" and block.name == "submit_move":
                    return dict(block.input)
            return None

        def validate_move(m):
            """Returns an error string if the move violates a physical rule, else None."""
            if m is None:
                return None
            limb = m.get("limb", "")
            hold = m.get("hold")

            # Hand must be in reachable list
            if limb == "right hand" and hold is not None:
                reachable_nums = [r["hold"] for r in graph.get(state["RH"], [])]
                if hold not in reachable_nums:
                    return f"Hold {hold} is not reachable from RH (Hold {state['RH']}). Choose a hold from the reachable list."
            if limb == "left hand" and hold is not None:
                reachable_nums = [r["hold"] for r in graph.get(state["LH"], [])]
                if hold not in reachable_nums:
                    return f"Hold {hold} is not reachable from LH (Hold {state['LH']}). Choose a hold from the reachable list."

            # Two feet cannot share a small chip
            if limb in ("right foot", "swap right foot") and hold is not None:
                other_foot = state["LF"]
                if other_foot == hold and hold_map.get(hold, {}).get("size", "").startswith("small"):
                    return f"Both feet cannot share Hold {hold} — it is a small chip. Use foot swap or choose a different hold."
            if limb in ("left foot", "swap left foot") and hold is not None:
                other_foot = state["RF"]
                if other_foot == hold and hold_map.get(hold, {}).get("size", "").startswith("small"):
                    return f"Both feet cannot share Hold {hold} — it is a small chip. Use foot swap or choose a different hold."

            # Hold must exist
            if hold is not None and hold not in hold_map:
                return f"Hold {hold} does not exist on this wall. Choose a valid hold number."

            # Over-extension: a hand cannot end up beyond full body extension
            # from the feet — at some point a foot has to move up
            if max_span_px and limb in ("right hand", "left hand", "both hands") and hold is not None:
                foot_holds = [state[f] for f in ("LF", "RF")
                              if state[f] is not None and state[f] in hold_map]
                if foot_holds:
                    hx, hy = hold_map[hold]["x"], hold_map[hold]["y"]
                    closest_foot = min(
                        ((hold_map[f]["x"] - hx) ** 2 + (hold_map[f]["y"] - hy) ** 2) ** 0.5
                        for f in foot_holds
                    )
                    if closest_foot > max_span_px:
                        return (f"Hold {hold} is beyond full body extension from your current feet. "
                                f"Move a foot up to a higher foothold first, then reach with the hand.")

            return None

        # Request move with up to 2 correction attempts
        move = request_move()
        for _ in range(2):
            error = validate_move(move)
            if error is None:
                break
            move = request_move(extra_correction=error)

        if move is None:
            break

        move["move_number"] = move_num

        # Final guard — skip invalid hold rather than crash
        if move["hold"] is not None and move["hold"] not in hold_map:
            break

        state = apply_move_to_state(state, move)
        sequence.append(move)
        states.append(dict(state))

        if move["action"] == "finish":
            break
        if move["hold"] == finish_hold_num and move["limb"] in ("right hand", "left hand", "both hands"):
            break

    return sequence, states


def format_sequence_as_text(sequence, hold_descriptions):
    lines = []
    for move in sequence:
        num      = move["move_number"]
        limb     = move["limb"]
        hold     = move["hold"]
        cue      = move.get("cue", "")
        hold_str = f"Hold {hold}" if hold is not None else "wall"

        if num == 0:
            lines.append(f"**Start:** {cue}")
        else:
            lines.append(f"**Move {num}:** {limb} → {hold_str} — {cue}")

    lines.append("\n**Hold analysis**\n")
    lines.append(hold_descriptions)
    return "\n\n".join(lines)


def analyze_user_beta(annotated_image, holds, user_beta, height_cm, difficulty, wall_angle):
    """Rate My Beta: the climber describes their own sequence and gets coaching feedback."""
    buf = io.BytesIO()
    annotated_image.save(buf, format="JPEG")
    image_data = base64.b64encode(buf.getvalue()).decode("utf-8")

    holds_description = build_holds_description(holds)

    response = anthropic_client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1500,
        system="""You are an experienced bouldering coach reviewing a climber's beta (their planned sequence). Be supportive but honest — like a good coach at the gym.

Your feedback should cover:
1. WHAT WORKS — parts of their beta that are solid, and why
2. WATCH OUT FOR — risks or inefficiencies (balance issues, skipped feet, over-gripping)
3. SUGGESTIONS — at most 2-3 concrete improvements, only where they genuinely help. If their beta is good, say so — do not invent problems.

Refer to holds by their numbers. Keep it conversational and under 300 words. Never rewrite their whole sequence — coach the beta they brought you.""",
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": image_data}},
                {"type": "text", "text": f"""Here is the climbing wall with numbered holds.

{holds_description}

CLIMBER: {height_cm}cm, {difficulty} level
WALL: {wall_angle}

MY BETA:
{user_beta}

Please review my beta."""}
            ]
        }]
    )
    return response.content[0].text


def draw_move_overlay(base_image, holds, sequence, current_step):
    hold_map = {h["number"]: h for h in holds}
    image = base_image.copy()
    draw  = ImageDraw.Draw(image)

    # One colour per limb, matching the animated figure:
    # LH yellow, RH orange, LF green, RF blue
    COLOURS = {
        "right hand":       "#fb923c",
        "left hand":        "#facc15",
        "both hands":       "#fb923c",
        "right foot":       "#38bdf8",
        "left foot":        "#4ade80",
        "both feet":        "#38bdf8",
        "flag right foot":  "#38bdf8",
        "flag left foot":   "#4ade80",
        "smear right foot": "#38bdf8",
        "smear left foot":  "#4ade80",
        "swap right foot":  "#38bdf8",
        "swap left foot":   "#4ade80",
    }

    for move in sequence[:current_step]:
        if move["hold"] is None:
            continue
        hold_num = move["hold"]
        if hold_num not in hold_map:
            continue
        hx, hy = hold_map[hold_num]["x"], hold_map[hold_num]["y"]
        colour = COLOURS.get(move["limb"], "#ffffff")
        draw.ellipse([hx-18, hy-18, hx+18, hy+18], outline=colour, width=2)

    current_move = sequence[current_step]
    if current_move["hold"] is not None:
        hold_num = current_move["hold"]
        if hold_num in hold_map:
            hx, hy = hold_map[hold_num]["x"], hold_map[hold_num]["y"]
            colour = COLOURS.get(current_move["limb"], "#ffffff")
            draw.ellipse([hx-35, hy-35, hx+35, hy+35], outline=colour, width=2)
            draw.ellipse([hx-25, hy-25, hx+25, hy+25], fill=colour)
            draw.text((hx-8, hy-10), str(current_move["move_number"]), fill="black")

    return image


# ---- UI ----

st.set_page_config(page_title="ClimbAI", page_icon="🧗", layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=DM+Sans:wght@400;500;600&display=swap');

:root {
    --bg: #0f1011;
    --surface: #1a1b1e;
    --border: #26282c;
    --text: #ececec;
    --muted: #9b9ea6;
    --accent: #fb923c;
}

html, body, [class*="st-"], .stMarkdown, input, textarea, select {
    font-family: 'DM Sans', -apple-system, sans-serif !important;
}

/* Restore Streamlit's icon font — icons are ligatures and render as raw words
   (e.g. "upload") if the font-family override reaches them */
[data-testid="stIconMaterial"], [class*="material-symbols"] {
    font-family: 'Material Symbols Rounded' !important;
}

/* Hide Streamlit chrome */
#MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; height: 0; }

/* Layout */
.block-container { max-width: 720px; padding-top: 4rem; padding-bottom: 6rem; }

/* Hero */
.climb-hero { margin-bottom: 3rem; }
.climb-hero .wordmark {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 3rem; font-weight: 700; letter-spacing: -0.045em;
    color: var(--text); margin: 0; line-height: 1;
}
.climb-hero .wordmark .accent { color: var(--accent); }
.climb-hero .tagline {
    color: var(--muted); font-size: 1.02rem; margin-top: 0.9rem;
    font-weight: 400; max-width: 30rem; line-height: 1.55;
}

/* Section labels — number, title, keyline */
.section-label {
    display: flex; align-items: center; gap: 0.75rem;
    margin: 3rem 0 0.3rem;
}
.section-label .num {
    font-family: 'Space Grotesk', sans-serif;
    color: var(--accent); font-size: 0.78rem; font-weight: 600; letter-spacing: 0.1em;
}
.section-label .title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.12rem; font-weight: 600; color: var(--text); letter-spacing: -0.01em;
    white-space: nowrap;
}
.section-label::after {
    content: ""; flex: 1; height: 1px; background: var(--border);
}
.section-caption { color: var(--muted); font-size: 0.87rem; margin: 0.15rem 0 0.7rem; line-height: 1.5; }
.accent-text { color: var(--accent); font-weight: 600; }

/* Buttons */
.stButton > button {
    font-family: 'Space Grotesk', sans-serif !important;
    border-radius: 10px; font-weight: 600; padding: 0.7rem 1rem;
    background: var(--surface); border: 1px solid var(--border); color: var(--text);
    transition: border-color 0.15s ease, color 0.15s ease, filter 0.15s ease;
}
.stButton > button:hover { border-color: var(--accent); color: var(--accent); }
.stButton > button[kind="primary"] {
    background: var(--accent); color: var(--bg); border: none;
}
.stButton > button[kind="primary"]:hover { filter: brightness(1.1); color: var(--bg); }

/* Inputs */
.stSelectbox > div > div, .stNumberInput > div > div, .stTextArea textarea {
    border-radius: 10px !important;
}
[data-testid="stFileUploaderDropzone"] {
    border-radius: 12px; border: 1px dashed var(--border) !important;
    background: var(--surface) !important;
}
/* Hide drag-and-drop instructions — single clean button, mobile friendly */
[data-testid="stFileUploaderDropzoneInstructions"] { display: none; }
[data-testid="stFileUploaderDropzone"] {
    justify-content: center; align-items: center; padding: 0.6rem; gap: 0;
}
/* The upload icon is a SIBLING of the button inside the dropzone —
   hide icons anywhere in the dropzone, at any depth */
[data-testid="stFileUploaderDropzone"] [data-testid="stIconMaterial"],
[data-testid="stFileUploaderDropzone"] [class*="material-symbols"],
[data-testid="stFileUploaderDropzone"] svg,
[data-testid="stFileUploaderDropzone"] i {
    display: none !important;
}
[data-testid="stFileUploaderDropzone"] button {
    font-family: 'Space Grotesk', sans-serif !important;
    border-radius: 8px; font-weight: 600;
    width: 100%; padding: 0.55rem 1rem;
    display: flex; align-items: center; justify-content: center; gap: 0;
    text-align: center;
}
[data-testid="stFileUploaderDropzone"] button > * { margin: 0 !important; }
/* The button sits inside an auto-width wrapper span — stretch it so the
   full-width button actually spans the dropzone */
[data-testid="stFileUploaderDropzone"] > span { display: block; width: 100%; }
[data-testid="stFileUploaderDropzone"] button * { margin: 0 !important; }

/* Alerts */
.stAlert { border-radius: 10px; }

/* Move card */
.move-card {
    background: var(--surface); border: 1px solid var(--border);
    border-left: 3px solid var(--accent); border-radius: 10px;
    padding: 0.95rem 1.2rem; margin: 0.4rem 0 1rem;
}
.move-card .move-num {
    font-family: 'Space Grotesk', sans-serif;
    color: var(--accent); font-weight: 600; font-size: 0.72rem;
    letter-spacing: 0.12em; text-transform: uppercase;
}
.move-card .move-text {
    font-family: 'Space Grotesk', sans-serif;
    color: var(--text); font-size: 1.05rem; margin-top: 4px; font-weight: 600;
}
.move-card .move-cue { color: var(--muted); font-size: 0.88rem; margin-top: 3px; }

/* Footer */
.climb-footer {
    font-family: 'Space Grotesk', sans-serif;
    color: #55585e; font-size: 0.72rem; text-align: center; margin-top: 5rem;
    text-transform: uppercase; letter-spacing: 0.18em;
}

hr { margin: 1.6rem 0 !important; opacity: 0.25; }
</style>

<div class="climb-hero">
    <h1 class="wordmark">climb<span class="accent">ai</span></h1>
    <p class="tagline">Route reading, assisted. Upload a wall photo — get a suggested beta and coaching feedback on your own.</p>
</div>
""", unsafe_allow_html=True)


def section_header(num, title, caption=None):
    st.markdown(f"""<div class="section-label"><span class="num">{num}</span><span class="title">{title}</span></div>""", unsafe_allow_html=True)
    if caption:
        st.markdown(f"""<p class="section-caption">{caption}</p>""", unsafe_allow_html=True)

section_header("01", "The route", "A clear, straight-on photo works best.")

uploaded_file = st.file_uploader("Climbing wall photo", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

col1, col2 = st.columns(2)
with col1:
    colour = st.selectbox("Route colour",
        ["Black", "Blue", "Red", "Green", "Orange", "Pink", "White", "Yellow", "Purple"])
with col2:
    wall_angle = st.selectbox("Wall angle",
        ["Slab (less than vertical)", "Vertical", "Slight overhang", "Steep overhang", "Roof"])

col3, col4 = st.columns(2)
with col3:
    start_style = st.selectbox("Start marking",
        ["START label", "Tape on hold", "Two hands on lowest holds", "Not marked"])
with col4:
    finish_style = st.selectbox("Finish marking",
        ["TOP label", "Tape on hold", "Top-out (both hands on top)", "Not marked"])

section_header("02", "About you")

col5, col6 = st.columns(2)
with col5:
    difficulty = st.selectbox("Experience level",
        ["Beginner", "Intermediate", "Advanced"])
with col6:
    height_cm = st.number_input("Height (cm)", min_value=140, max_value=220, value=170)

extra_notes = st.text_area("Notes (optional)",
    placeholder="e.g. 'There is a big move in the middle'",
    height=80)

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    section_header("03", "Detect holds")

    if st.button("Detect holds", use_container_width=True, type="primary"):
        with st.spinner("Detecting holds and analysing types..."):
            annotated_image, holds = detect_and_validate_holds(tmp_path, colour)
            st.session_state["annotated_image"] = annotated_image
            st.session_state["holds"]           = holds
            st.session_state["tmp_path"]        = tmp_path
            st.session_state["start_holds"]     = []
            st.session_state.pop("sequence",      None)
            st.session_state.pop("instructions",  None)
            st.session_state.pop("beta_feedback", None)
            st.session_state.pop("states",        None)
            st.session_state.pop("beta_gif",      None)
            st.session_state["current_step"]    = 0

    if "annotated_image" in st.session_state and st.session_state.get("holds"):
        holds          = st.session_state["holds"]
        annotated_image = st.session_state["annotated_image"]

        st.markdown(f"""<p class="section-caption" style="margin-top:0.8rem;">Found <span class="accent-text">{len(holds)}</span> {colour.lower()} holds. Click the start hold(s) below — one if both hands start together, two if they start apart. Click again to deselect.</p>""", unsafe_allow_html=True)

        from streamlit_image_coordinates import streamlit_image_coordinates
        value = streamlit_image_coordinates(annotated_image, key="hold_click")

        if value is not None and "sequence" not in st.session_state:
            click_x, click_y = value["x"], value["y"]
            closest_hold = None
            closest_dist = float("inf")
            for h in holds:
                dist = ((h["x"] - click_x) ** 2 + (h["y"] - click_y) ** 2) ** 0.5
                if dist < closest_dist:
                    closest_dist = dist
                    closest_hold = h

            if closest_hold and closest_dist < 40:
                last_click = st.session_state.get("last_click", None)
                new_click  = (value["x"], value["y"])

                if last_click != new_click:
                    st.session_state["last_click"] = new_click
                    start_holds = st.session_state.get("start_holds", [])
                    hold_nums   = [h["number"] for h in start_holds]
                    if closest_hold["number"] in hold_nums:
                        start_holds = [h for h in start_holds if h["number"] != closest_hold["number"]]
                        st.toast(f"Deselected Hold {closest_hold['number']}")
                    else:
                        start_holds.append(closest_hold)
                        st.toast(f"Selected Hold {closest_hold['number']} as start hold!")
                    st.session_state["start_holds"] = start_holds

        current_start_holds = st.session_state.get("start_holds", [])
        if current_start_holds:
            hold_nums = [str(h["number"]) for h in current_start_holds]
            st.success(f"Start holds: Hold {', Hold '.join(hold_nums)}")
        else:
            st.info("Click the start hold(s) on the image above.")

        section_header("04", "Suggested beta")

        if st.button("Generate suggested beta", use_container_width=True, type="primary"):
            current_start_holds = st.session_state.get("start_holds", [])
            if not current_start_holds:
                st.warning("Please click on the start holds in the image before generating instructions.")
            else:
                image = Image.open(st.session_state["tmp_path"])
                image_width, image_height = image.size

                graph = build_reachability_graph(
                    st.session_state["holds"],
                    image_height, image_width, height_cm
                )

                hold_descriptions = build_holds_description(st.session_state["holds"])

                progress_bar = st.progress(0)
                status       = st.empty()

                def on_progress(move_num, max_moves):
                    progress_bar.progress(move_num / max_moves)
                    status.caption(f"Generating move {move_num}...")

                sequence, states = generate_sequence_iteratively(
                    hold_descriptions,
                    st.session_state["holds"],
                    graph,
                    current_start_holds,
                    height_cm, difficulty, wall_angle, finish_style, extra_notes,
                    image_height=image_height,
                    progress_callback=on_progress
                )

                progress_bar.empty()
                status.empty()

                instructions = format_sequence_as_text(sequence, hold_descriptions)
                st.session_state["instructions"] = instructions
                st.session_state["sequence"]     = sequence
                st.session_state["states"]       = states
                st.session_state["current_step"] = 0
                st.session_state["base_image"]   = st.session_state["annotated_image"]
                st.session_state.pop("beta_gif", None)

    if "instructions" in st.session_state:
        st.markdown("""<p class="section-caption" style="margin-top:0.8rem;">A starting point — adapt it to your body, strengths, and style. Even experienced climbers refine beta on the wall.</p>""", unsafe_allow_html=True)
        st.markdown(st.session_state["instructions"])

    if "sequence" in st.session_state and st.session_state["sequence"]:
        sequence     = st.session_state["sequence"]
        current_step = st.session_state.get("current_step", 0)

        section_header("05", "Walk it through", "Step through each move on the wall.")

        overlay_image = draw_move_overlay(
            st.session_state["base_image"],
            st.session_state["holds"],
            sequence,
            current_step
        )

        current_move = sequence[current_step]
        limb = current_move["limb"]
        hold = current_move["hold"]
        cue  = current_move.get("cue", "")

        hold_str = f"Hold {hold}" if hold is not None else "wall"
        st.markdown(f"""
        <div class="move-card">
            <div class="move-num">Move {current_step + 1} of {len(sequence)}</div>
            <div class="move-text">{limb} &rarr; {hold_str}</div>
            <div class="move-cue">{cue}</div>
        </div>
        """, unsafe_allow_html=True)

        st.image(overlay_image, width=700)

        col_prev, col_next = st.columns(2)
        with col_prev:
            if st.button("Previous", use_container_width=True):
                if st.session_state["current_step"] > 0:
                    st.session_state["current_step"] -= 1
                    st.rerun()
        with col_next:
            if st.button("Next", use_container_width=True):
                if st.session_state["current_step"] < len(sequence) - 1:
                    st.session_state["current_step"] += 1
                    st.rerun()

        # ---- Animated demo ----
        if "states" in st.session_state:
            st.markdown("""<p class="section-caption" style="margin-top:1.2rem;">Or watch the whole beta as an animation.</p>""", unsafe_allow_html=True)

            if "beta_gif" not in st.session_state:
                if st.button("🎬 Animate the beta", use_container_width=True):
                    with st.spinner("Rendering animation..."):
                        gif_buf = render_beta_gif(
                            st.session_state["base_image"],
                            st.session_state["holds"],
                            sequence,
                            st.session_state["states"],
                        )
                        st.session_state["beta_gif"] = gif_buf.getvalue()
                    st.rerun()

            if "beta_gif" in st.session_state:
                st.image(st.session_state["beta_gif"])
                st.download_button(
                    "Download GIF",
                    st.session_state["beta_gif"],
                    file_name="climbai-beta.gif",
                    mime="image/gif",
                    use_container_width=True,
                )

    # ---- Rate My Beta ----
    if "annotated_image" in st.session_state and st.session_state.get("holds"):
        section_header("06", "Rate my beta", "Already have a sequence in mind? Describe it and get coaching feedback.")

        user_beta = st.text_area(
            "Describe your beta",
            placeholder="e.g. Start both hands on hold 5, LF on 2, RF on 1. Step RF up to 3, RH to 6, swap feet on 3...",
            height=150,
            key="user_beta_input",
            label_visibility="collapsed"
        )

        if st.button("Get coaching feedback", use_container_width=True, type="primary"):
            if not user_beta.strip():
                st.warning("Describe your sequence first — which hands and feet go where, in order.")
            else:
                with st.spinner("Your coach is taking a look..."):
                    feedback = analyze_user_beta(
                        st.session_state["annotated_image"],
                        st.session_state["holds"],
                        user_beta,
                        height_cm, difficulty, wall_angle
                    )
                    st.session_state["beta_feedback"] = feedback

        if "beta_feedback" in st.session_state:
            st.markdown(st.session_state["beta_feedback"])

st.markdown("""<p class="climb-footer">climbai &mdash; get through the plateau</p>""", unsafe_allow_html=True)
