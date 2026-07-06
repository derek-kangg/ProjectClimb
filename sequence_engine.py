"""Headless beta-generation engine.

All the sequence logic lives here, independent of Streamlit, so the same
code drives the web app, the eval harness (scripts/eval_route.py), and a
future API backend. Python owns the body state; the model only ever picks
the next single move, and every move passes physical validation before it
is accepted.
"""

from climbing_knowledge import technique_library

GENERATION_MODEL = "claude-sonnet-4-6"


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
        orient    = h.get("orientation", "unknown")
        orient_str = f", orientation={orient}" if orient not in ("unknown", "top") else ""
        lines.append(
            f"Hold {h['number']}: {hold_type}, {size}, {side} side, best_use={best_use}{orient_str}, x={h['x']}, y={h['y']}"
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
    client, hold_descriptions, holds, graph, start_holds,
    height_cm, difficulty, wall_angle, finish_style, extra_notes,
    image_height=None, progress_callback=None
):
    """
    Generate a route sequence one move at a time.
    Python tracks body state; the model only picks the next single move.
    `client` is an anthropic.Anthropic instance.
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
        + (f", ORIENTATION={h['orientation']}" if h.get("orientation") not in (None, "unknown", "top") else "")
        for h in sorted(holds, key=lambda x: x["y"], reverse=True)
    )

    move_tool = {
        "name": "submit_move",
        "description": "Submit the single best next climbing move given the current body state.",
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
                    "enum": ["move", "match", "flag", "smear", "swap", "heel hook", "toe hook", "drop knee", "knee bar", "backstep", "rockover", "deadpoint", "dyno", "mantle", "finish"]
                },
                "cue": {
                    "type": "string",
                    "description": "One short technique cue under 10 words"
                }
            },
            "required": ["limb", "hold", "action", "cue"]
        }
    }

    # Expert technique + grip knowledge, scaled to level and wall angle
    technique_section = technique_library(difficulty, wall_angle)

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

        prompt = f"""You are an experienced climbing coach generating a bouldering sequence one move at a time.

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
            resp = client.messages.create(
                model=GENERATION_MODEL,
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

    # A boulder is finished with BOTH hands matched on the finish hold —
    # if only one hand got there, append the match deterministically.
    lh_on = state["LH"] == finish_hold_num
    rh_on = state["RH"] == finish_hold_num
    if lh_on != rh_on:
        match_move = {
            "move_number": sequence[-1]["move_number"] + 1,
            "limb": "left hand" if rh_on else "right hand",
            "hold": finish_hold_num,
            "action": "match",
            "cue": "Match hands on the finish hold and control it",
        }
        state = apply_move_to_state(state, match_move)
        sequence.append(match_move)
        states.append(dict(state))

    return sequence, states
