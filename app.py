import streamlit as st
from openai import OpenAI
from PIL import Image, ImageDraw
from dotenv import load_dotenv
import cv2
import numpy as np
import os
import base64
import tempfile

load_dotenv()

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

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

def detect_holds_by_colour(image_path, colour, min_area=80):
    cv_image = cv2.imread(image_path)
    hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

    low, high = COLOUR_RANGES[colour]
    mask = cv2.inRange(hsv, np.array(low), np.array(high))

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    pil_image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(pil_image)

    holds = []
    hold_number = 1

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue

        x, y, w, h = cv2.boundingRect(contour)

        if w < 12 or h < 12:
            continue

        if y < cv_image.shape[0] * 0.03:
            continue

        cx, cy = x + w // 2, y + h // 2

        holds.append({"number": hold_number, "x": cx, "y": cy})

        draw.rectangle([x, y, x + w, y + h], outline="#6fcf4a", width=3)
        draw.ellipse([cx-15, cy-15, cx+15, cy+15], fill="#6fcf4a")
        draw.text((cx-5, cy-8), str(hold_number), fill="black")
        hold_number += 1

    return pil_image, holds

def build_holds_description(holds):
    if not holds:
        return "No holds detected."
    sorted_holds = sorted(holds, key=lambda h: h["y"], reverse=True)
    lines = ["Holds listed from bottom to top of the wall:"]
    for h in sorted_holds:
        lines.append(f"Hold {h['number']}: position x={h['x']}, y={h['y']}")
    return "\n".join(lines)

def get_route_instructions(image_path, colour, difficulty, height_cm, holds, wall_angle, start_style, finish_style, extra_notes, start_holds):
    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode("utf-8")

    holds_description = build_holds_description(holds)
    hold_numbers = [str(h["number"]) for h in holds]
    hold_list = ", ".join(hold_numbers)

    angle_context = {
        "Slab (less than vertical)": "This is a SLAB wall (less than vertical). Key advice: weight must be over feet at all times, trust your shoes, stand tall and avoid pulling with arms, balance and precise footwork are everything. Slipping off a slab usually means your weight shifted back.",
        "Vertical": "This is a VERTICAL wall. Key advice: keep arms straight to conserve energy, balance arm and leg use equally, look for resting positions where you can shake out.",
        "Slight overhang": "This is a SLIGHT OVERHANG. Key advice: keep hips close to the wall, start thinking about grip endurance, use momentum where possible, feet are still very important.",
        "Steep overhang": "This is a STEEP OVERHANG. Key advice: body tension is critical, keep feet on the wall at all times, move efficiently and quickly to conserve grip strength, technique matters more than raw strength here.",
        "Roof": "This is a ROOF (near horizontal). Key advice: maximum body tension required, heel hooks and toe hooks are essential, every move costs significant energy so plan ahead, core strength is critical."
    }

    start_context = {
        "START label": "The starting holds are marked with a START label.",
        "Tape on hold": "The starting holds are marked with tape. Look for taped holds at the bottom of the route.",
        "Two hands on lowest holds": "There is no start marker — the climber begins with both hands on the two lowest holds of the route.",
        "Not marked": "There is no start marker — use the lowest holds of the route to begin."
    }

    finish_context = {
        "TOP label": "The finishing hold is marked with a TOP label.",
        "Tape on hold": "The finishing hold is marked with tape.",
        "Top-out (both hands on top)": "To finish, the climber must get both hands on the very top of the wall.",
        "Not marked": "There is no finish marker — the route ends at the highest hold of the colour."
    }

    extra = f"\nExtra notes from the climber: {extra_notes}" if extra_notes.strip() else ""

    start_hold_nums = [str(h["number"]) for h in start_holds]
    if len(start_holds) == 1:
        start_holds_text = f"The climber has confirmed the START hold is: Hold {start_hold_nums[0]}. The climber begins with BOTH hands on this single hold."
    else:
        start_holds_text = f"The climber has confirmed the START holds are: Hold {', Hold '.join(start_hold_nums)}. The climber begins with one hand on each hold."

    message = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}
                },
                {
                    "type": "text",
                    "text": f"""You are an expert rock climbing coach analyzing a bouldering wall photo.

CLIMBER INFO:
- Height: {height_cm}cm
- Experience: {difficulty}
- Max reach between holds: approximately {int(height_cm * 1.3)}cm

WALL INFO:
- {angle_context[wall_angle]}
- Start: {start_holds_text}
- Finish: {finish_context[finish_style]}
{extra}

DETECTED {colour.upper()} HOLDS (numbered on the image):
{holds_description}
Hold numbers present: {hold_list}

INSTRUCTIONS:
As you analyze the image, also identify the types of holds you see (crimps, jugs, slopers, pinches, pockets etc) and factor this into your advice.

Provide a route breakdown with:

1. **Route Overview:** Difficulty, style, and what makes this route challenging or accessible for a {height_cm}cm {difficulty} climber on a {wall_angle} wall.

2. **Hold Types Identified:** List the hold types you can see and briefly explain how to grip each one.

3. **Starting Position:** Based on the confirmed start holds, describe exact hand and foot placement using hold numbers.

4. **Step by Step Moves:** For each move:
   - Which hold number, which hand or foot
   - Only suggest moves to realistically reachable holds
   - Body positioning for a {wall_angle} wall specifically
   - Technique with simple explanation in brackets
   - Height note if relevant for this specific move
   - Difficulty: Easy / Medium / Hard

5. **Finishing Move:** How to reach and complete the finish based on ({finish_style}).

6. **Key Tips:** 3 tips specific to this route, this wall angle, and this climber.

Always reference holds by number. Tailor ALL advice to the {wall_angle} wall type."""
                }
            ]
        }]
    )
    return message.choices[0].message.content


# ---- UI ----

st.set_page_config(page_title="ClimbAI", page_icon="🧗", layout="centered")

st.markdown("""
    <h1 style='text-align: center; color: #6fcf4a; font-size: 3rem;'>🧗 ClimbAI</h1>
    <p style='text-align: center; color: #6b806b; margin-bottom: 2rem;'>
        Upload a photo of a climbing wall and get an expert route breakdown
    </p>
""", unsafe_allow_html=True)

st.divider()

col1, col2 = st.columns([2, 1])

with col1:
    uploaded_file = st.file_uploader("Upload a climbing wall photo", type=["jpg", "jpeg", "png"])

with col2:
    colour = st.selectbox("Route colour",
        ["Black", "Blue", "Red", "Green", "Orange", "Pink", "White", "Yellow", "Purple"])
    difficulty = st.selectbox("Your experience level",
        ["Beginner", "Intermediate", "Advanced"])
    height_cm = st.number_input("Your height (cm)", min_value=140, max_value=220, value=170)

st.divider()
st.markdown("#### Wall & Route Details")

col3, col4 = st.columns(2)

with col3:
    wall_angle = st.selectbox("Wall angle",
        ["Slab (less than vertical)", "Vertical", "Slight overhang", "Steep overhang", "Roof"])
    start_style = st.selectbox("How is the start marked?",
        ["START label", "Tape on hold", "Two hands on lowest holds", "Not marked"])

with col4:
    finish_style = st.selectbox("How is the finish marked?",
        ["TOP label", "Tape on hold", "Top-out (both hands on top)", "Not marked"])
    extra_notes = st.text_area("Any extra notes about this route?",
        placeholder="e.g. 'There is a big move in the middle' or 'The gym is Movement SFU'",
        height=100)

if uploaded_file is not None:

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    st.divider()

    if st.button("🔍 Detect Holds", use_container_width=True):
        with st.spinner("Detecting holds..."):
            annotated_image, holds = detect_holds_by_colour(tmp_path, colour)
            st.session_state["annotated_image"] = annotated_image
            st.session_state["holds"] = holds
            st.session_state["tmp_path"] = tmp_path
            st.session_state["start_holds"] = []

    if "annotated_image" in st.session_state and st.session_state["holds"]:
        holds = st.session_state["holds"]
        annotated_image = st.session_state["annotated_image"]

        st.markdown(f"### 🎯 Detected {len(holds)} {colour} holds")

        st.markdown("""
            <p style='color: #6b806b; font-size: 0.85rem;'>
            👇 Click on the <b>start hold(s)</b> in the image below. Select <b>one hold</b> if both hands start on the same hold, or <b>two holds</b> if each hand starts on a different hold. Click again to deselect.
            </p>
        """, unsafe_allow_html=True)

        from streamlit_image_coordinates import streamlit_image_coordinates

        value = streamlit_image_coordinates(annotated_image, key="hold_click")

        if value is not None:
            click_x, click_y = value["x"], value["y"]

            closest_hold = None
            closest_dist = float("inf")
            for h in holds:
                dist = ((h["x"] - click_x) ** 2 + (h["y"] - click_y) ** 2) ** 0.5
                if dist < closest_dist:
                    closest_dist = dist
                    closest_hold = h

            if closest_hold and closest_dist < 40:
                start_holds = st.session_state.get("start_holds", [])
                hold_nums = [h["number"] for h in start_holds]

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
            st.success(f"✅ Start holds selected: Hold {', Hold '.join(hold_nums)}")
        else:
            st.info("No start holds selected yet — click the start holds on the image above.")

        st.divider()

        if st.button("📋 Generate Route Instructions", use_container_width=True):
            current_start_holds = st.session_state.get("start_holds", [])
        if not current_start_holds:
            st.warning("Please click on the start holds in the image before generating instructions.")
        else:
            with st.spinner("Generating route instructions..."):
                instructions = get_route_instructions(
                st.session_state["tmp_path"], colour, difficulty, height_cm,
                st.session_state["holds"], wall_angle, start_style, finish_style, extra_notes,
                current_start_holds
            )

            st.markdown("### 📋 Route Breakdown")
            st.markdown(instructions)
            st.divider()
            st.caption("ClimbAI — helping climbers of all levels get through plateaus 🧗")