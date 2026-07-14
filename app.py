from route_graph import build_reachability_graph, format_graph_for_prompt
from beta_animation import render_beta_gif
from climbing_knowledge import coaching_knowledge
from sequence_engine import build_holds_description, generate_sequence_iteratively
from hold_detection import find_hold_candidates
from PIL import Image, ImageDraw, ImageFont
import streamlit as st
import anthropic
from dotenv import load_dotenv
import os
import base64
import tempfile
import io
import json
import shutil
import re

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

_APP_DIR = os.path.dirname(os.path.abspath(__file__))
ROUTES_ROOT      = os.path.join(_APP_DIR, "saved_routes")
TEST_ROUTES_ROOT = os.path.join(_APP_DIR, "test_routes")
SAVE_DIR         = os.path.join(ROUTES_ROOT, "last")


def _slugify(name):
    return re.sub(r"[^a-z0-9]+", "-", (name or "").lower()).strip("-")


def save_route(target_dir=None):
    """Persist the current route so it can be resumed later without
    re-running hold detection or beta generation (no API cost)."""
    target = target_dir or SAVE_DIR
    os.makedirs(target, exist_ok=True)
    ss = st.session_state
    data = {
        "holds":        ss.get("holds"),
        "start_holds":  ss.get("start_holds", []),
        "sequence":     ss.get("sequence"),
        "states":       ss.get("states"),
        "instructions": ss.get("instructions"),
    }
    with open(os.path.join(target, "route.json"), "w", encoding="utf-8") as f:
        json.dump(data, f)
    if ss.get("annotated_image") is not None:
        ss["annotated_image"].save(os.path.join(target, "annotated.png"))
    tmp = ss.get("tmp_path")
    original = os.path.join(target, "original.jpg")
    if tmp and os.path.exists(tmp) and os.path.abspath(tmp) != os.path.abspath(original):
        shutil.copyfile(tmp, original)
    if ss.get("beta_gif"):
        with open(os.path.join(target, "beta.gif"), "wb") as f:
            f.write(ss["beta_gif"])


def list_saved_routes():
    """Names of all saved routes, 'last' (autosave) first."""
    if not os.path.isdir(ROUTES_ROOT):
        return []
    names = [n for n in sorted(os.listdir(ROUTES_ROOT))
             if os.path.exists(os.path.join(ROUTES_ROOT, n, "route.json"))]
    if "last" in names:
        names.remove("last")
        names.insert(0, "last")
    return names


def load_route(source_dir=None):
    """Restore a saved route into session state. Returns True on success."""
    source = source_dir or SAVE_DIR
    try:
        with open(os.path.join(source, "route.json"), "r", encoding="utf-8") as f:
            data = json.load(f)
        annotated = Image.open(os.path.join(source, "annotated.png")).convert("RGB")
    except Exception:
        return False
    ss = st.session_state
    ss["holds"]           = data.get("holds") or []
    ss["start_holds"]     = data.get("start_holds") or []
    ss["annotated_image"] = annotated
    ss["base_image"]      = annotated
    ss["tmp_path"]        = os.path.join(source, "original.jpg")
    ss["current_step"]    = 0
    ss.pop("sequence", None)
    ss.pop("states", None)
    ss.pop("instructions", None)
    ss.pop("beta_gif", None)
    if data.get("sequence"):
        ss["sequence"] = data["sequence"]
    if data.get("states"):
        ss["states"] = data["states"]
    if data.get("instructions"):
        ss["instructions"] = data["instructions"]
    gif_path = os.path.join(source, "beta.gif")
    if os.path.exists(gif_path):
        with open(gif_path, "rb") as f:
            ss["beta_gif"] = f.read()
    return True


def export_test_route(slug, colour, wall_angle, difficulty):
    """Write the current route into test_routes/<slug>/ in the evaluation
    format: wall photo, machine-readable data, and a route.md with the AI
    beta prefilled and a section for the climber's verified beta."""
    target = os.path.join(TEST_ROUTES_ROOT, slug)
    os.makedirs(target, exist_ok=True)
    ss = st.session_state

    tmp = ss.get("tmp_path")
    if tmp and os.path.exists(tmp):
        shutil.copyfile(tmp, os.path.join(target, "wall.jpg"))

    holds = ss.get("holds") or []
    seq   = ss.get("sequence") or []

    hold_lines = "\n".join(
        f"- Hold {h['number']}: {h.get('hold_type', '?')}, {h.get('size', '?')}"
        + (f", {h['orientation']}" if h.get("orientation") not in (None, "unknown", "top") else "")
        + f", x={h['x']}, y={h['y']}"
        for h in holds
    )
    ai_lines = "\n".join(
        f"{m['move_number']}. {m['limb']} -> "
        + (f"hold {m['hold']}" if m["hold"] is not None else "wall")
        + f" ({m['action']}) - {m.get('cue', '')}"
        for m in seq
    )

    md = f"""# {slug}

- Gym / wall:
- Colour: {colour}
- Grade:
- Wall angle: {wall_angle}
- Level used in app: {difficulty}
- Start:
- Finish:

## Detected holds

{hold_lines}

## AI suggested beta (for reference)

{ai_lines}

## My verified beta (fill this in after climbing)

START:
1.

## Notes

"""
    with open(os.path.join(target, "route.md"), "w", encoding="utf-8") as f:
        f.write(md)
    with open(os.path.join(target, "route_data.json"), "w", encoding="utf-8") as f:
        json.dump({
            "holds": holds, "ai_sequence": seq, "colour": colour,
            "wall_angle": wall_angle, "difficulty": difficulty,
        }, f, indent=2)
    return os.path.relpath(target, _APP_DIR)

def _marker_metrics(img_width):
    """Marker radius and font scaled to image resolution — phone photos are
    ~2000px+ and fixed-size markers become unreadably small."""
    r = max(15, img_width // 70)
    try:
        font = ImageFont.load_default(size=max(11, int(r * 1.1)))
    except TypeError:
        font = ImageFont.load_default()
    return r, font


def detect_and_validate_holds(image_path, colour):
    """
    Two-pass hold detection:
    Pass 1 — OpenCV finds candidates by colour (hold_detection module:
              colour mask + merged-blob splitting + tape filter).
    Pass 2 — Claude Sonnet sees the annotated image and labels each hold's
              type, best use, and orientation in one vision call.
    Returns: (annotated PIL image, holds list)
    """
    candidates = find_hold_candidates(image_path, colour)

    if not candidates:
        return Image.open(image_path).convert("RGB"), []

    # Draw numbered candidates on a preview image for Claude to assess
    preview = Image.open(image_path).convert("RGB")
    preview_draw = ImageDraw.Draw(preview)
    mr, mfont = _marker_metrics(preview.width)
    for i, c in enumerate(candidates, 1):
        cx, cy = c["x"], c["y"]
        preview_draw.ellipse([cx-mr, cy-mr, cx+mr, cy+mr], fill="#fb923c")
        preview_draw.text((cx, cy), str(i), fill="black", font=mfont, anchor="mm")

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
                            "number":      {"type": "integer"},
                            "hold_type":   {"type": "string", "enum": ["jug", "crimp", "sloper", "pinch", "pocket", "edge", "volume", "chip", "unknown"]},
                            "best_use":    {"type": "string", "enum": ["handhold", "foothold", "both"]},
                            "orientation": {"type": "string", "enum": ["top", "undercling", "side-pull left", "side-pull right", "unknown"], "description": "Which direction the usable surface faces: top = pull down on it (normal), undercling = usable surface faces down so palm goes up, side-pull left/right = vertical edge pulled sideways (side of the usable surface)"}
                        },
                        "required": ["number", "hold_type", "best_use", "orientation"]
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
3. Orientation — which way the usable surface faces. This changes how the hold is climbed: top (pull down, normal), undercling (usable surface faces DOWN, palm-up grip), side-pull left or right (vertical edge pulled sideways). Look carefully at the shadows and shape.

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
    mr, mfont = _marker_metrics(final_image.width)
    box_w = max(3, final_image.width // 500)
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
            "number":      new_number,
            "x":           cx,
            "y":           cy,
            "size":        size_label,
            "hold_type":   hold_type,
            "best_use":    best_use,
            "orientation": v.get("orientation", "unknown"),
        })

        draw.rectangle([bx, by, bx + bw, by + bh], outline="#fb923c", width=box_w)
        draw.ellipse([cx-mr, cy-mr, cx+mr, cy+mr], fill="#fb923c")
        draw.text((cx, cy), str(new_number), fill="black", font=mfont, anchor="mm")
        new_number += 1

    return final_image, final_holds


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
        system=f"""You are an experienced bouldering coach reviewing a climber's beta (their planned sequence). Be supportive but honest — like a good coach at the gym.

Your feedback should cover:
1. WHAT WORKS — parts of their beta that are solid, and why
2. WATCH OUT FOR — risks or inefficiencies (balance issues, skipped feet, over-gripping, wrong grip for the hold type)
3. SUGGESTIONS — at most 2-3 concrete improvements, only where they genuinely help. If their beta is good, say so — do not invent problems.

Refer to holds by their numbers. Use precise technique vocabulary where it fits (grip choice for the hold type, drop knee vs backstep, deadpoint vs dyno). Keep it conversational and under 300 words. Never rewrite their whole sequence — coach the beta they brought you.

{coaching_knowledge()}""",
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

    # marker sizes scale with resolution (fixed sizes vanish on phone photos)
    s = max(1.0, image.width / 770.0)
    ring_w = max(2, int(2 * s))
    mr, mfont = _marker_metrics(image.width)

    for move in sequence[:current_step]:
        if move["hold"] is None:
            continue
        hold_num = move["hold"]
        if hold_num not in hold_map:
            continue
        hx, hy = hold_map[hold_num]["x"], hold_map[hold_num]["y"]
        colour = COLOURS.get(move["limb"], "#ffffff")
        r = int(18 * s)
        draw.ellipse([hx-r, hy-r, hx+r, hy+r], outline=colour, width=ring_w)

    current_move = sequence[current_step]
    if current_move["hold"] is not None:
        hold_num = current_move["hold"]
        if hold_num in hold_map:
            hx, hy = hold_map[hold_num]["x"], hold_map[hold_num]["y"]
            colour = COLOURS.get(current_move["limb"], "#ffffff")
            r_out, r_in = int(35 * s), int(25 * s)
            draw.ellipse([hx-r_out, hy-r_out, hx+r_out, hy+r_out], outline=colour, width=ring_w)
            draw.ellipse([hx-r_in, hy-r_in, hx+r_in, hy+r_in], fill=colour)
            draw.text((hx, hy), str(current_move["move_number"]), fill="black", font=mfont, anchor="mm")

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
        try:
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
                save_route()
        except Exception:
            st.error("Hold detection hit a snag — likely a network hiccup. Give it another go.")

elif "holds" not in st.session_state and list_saved_routes():
    section_header("03", "Detect holds")
    st.markdown("""<p class="section-caption">No photo uploaded — load a route from your library instead. No re-detection or regeneration needed.</p>""", unsafe_allow_html=True)
    col_pick, col_load = st.columns([2, 1])
    with col_pick:
        chosen_route = st.selectbox("Saved routes", list_saved_routes(), label_visibility="collapsed")
    with col_load:
        if st.button("Load route", use_container_width=True, type="primary"):
            if load_route(os.path.join(ROUTES_ROOT, chosen_route)):
                st.rerun()
            else:
                st.warning("Couldn't load that route — upload a photo instead.")

if "annotated_image" in st.session_state and st.session_state.get("holds"):
    holds           = st.session_state["holds"]
    annotated_image = st.session_state["annotated_image"]

    st.markdown(f"""<p class="section-caption" style="margin-top:0.8rem;">Found <span class="accent-text">{len(holds)}</span> {colour.lower()} holds. Click the start hold(s) below — one if both hands start together, two if they start apart. Click again to deselect.</p>""", unsafe_allow_html=True)

    from streamlit_image_coordinates import streamlit_image_coordinates
    value = streamlit_image_coordinates(annotated_image, key="hold_click", use_column_width="always")

    if value is not None and "sequence" not in st.session_state:
        # The component reports clicks in DISPLAYED pixels (plus the displayed
        # size) — rescale into original image space or taps on phones, where
        # the image is shrunk to fit, would land on the wrong holds.
        disp_w = value.get("width") or annotated_image.width
        disp_h = value.get("height") or annotated_image.height
        click_x = value["x"] * annotated_image.width / disp_w
        click_y = value["y"] * annotated_image.height / disp_h

        # Match threshold scales with image size (fat-finger friendly)
        threshold = max(40, annotated_image.width * 0.06)

        closest_hold = None
        closest_dist = float("inf")
        for h in holds:
            dist = ((h["x"] - click_x) ** 2 + (h["y"] - click_y) ** 2) ** 0.5
            if dist < closest_dist:
                closest_dist = dist
                closest_hold = h

        if closest_hold and closest_dist < threshold:
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

            try:
                sequence, states = generate_sequence_iteratively(
                    anthropic_client,
                    hold_descriptions,
                    st.session_state["holds"],
                    graph,
                    current_start_holds,
                    height_cm, difficulty, wall_angle, finish_style, extra_notes,
                    image_height=image_height,
                    progress_callback=on_progress
                )
            except Exception:
                progress_bar.empty()
                status.empty()
                st.error("Beta generation was interrupted — likely a network hiccup. Your holds are still here; just hit generate again.")
            else:
                progress_bar.empty()
                status.empty()

                instructions = format_sequence_as_text(sequence, hold_descriptions)
                st.session_state["instructions"] = instructions
                st.session_state["sequence"]     = sequence
                st.session_state["states"]       = states
                st.session_state["current_step"] = 0
                st.session_state["base_image"]   = st.session_state["annotated_image"]
                st.session_state.pop("beta_gif", None)
                save_route()

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

        def _render_and_store_gif():
            with st.spinner("Rendering animation..."):
                gif_buf = render_beta_gif(
                    st.session_state["base_image"],
                    st.session_state["holds"],
                    sequence,
                    st.session_state["states"],
                    height_cm=height_cm,
                )
                st.session_state["beta_gif"] = gif_buf.getvalue()
                save_route()

        if "beta_gif" not in st.session_state:
            if st.button("🎬 Animate the beta", use_container_width=True):
                _render_and_store_gif()
                st.rerun()

        if "beta_gif" in st.session_state:
            st.image(st.session_state["beta_gif"])
            col_re, col_dl = st.columns(2)
            with col_re:
                if st.button("Re-render animation", use_container_width=True):
                    _render_and_store_gif()
                    st.rerun()
            with col_dl:
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
            try:
                with st.spinner("Your coach is taking a look..."):
                    feedback = analyze_user_beta(
                        st.session_state["annotated_image"],
                        st.session_state["holds"],
                        user_beta,
                        height_cm, difficulty, wall_angle
                    )
                    st.session_state["beta_feedback"] = feedback
            except Exception:
                st.error("The coach lost connection — try again in a moment.")

    if "beta_feedback" in st.session_state:
        st.markdown(st.session_state["beta_feedback"])

# ---- Save & export ----
if "sequence" in st.session_state and st.session_state.get("holds"):
    section_header("07", "Save this route", "Keep it in your library, or export it as a test case for improving the AI.")

    route_name = st.text_input(
        "Route name",
        placeholder="e.g. movement-blue-v2",
        key="route_name",
        label_visibility="collapsed"
    )

    col_save, col_export = st.columns(2)
    with col_save:
        if st.button("Save to library", use_container_width=True):
            slug = _slugify(route_name)
            if not slug:
                st.warning("Give the route a name first.")
            else:
                save_route(os.path.join(ROUTES_ROOT, slug))
                st.success(f"Saved to library as '{slug}' — load it any time from the start screen.")
    with col_export:
        if st.button("Export as test route", use_container_width=True):
            slug = _slugify(route_name)
            if not slug:
                st.warning("Give the route a name first.")
            else:
                path = export_test_route(slug, colour, wall_angle, difficulty)
                st.success(f"Exported to {path} — open route.md and fill in your verified beta after climbing it.")

st.markdown("""<p class="climb-footer">climbai &mdash; get through the plateau</p>""", unsafe_allow_html=True)
