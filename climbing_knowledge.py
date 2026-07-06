"""Expert climbing knowledge injected into the AI prompts.

One source of truth for grips, techniques, and coaching wisdom so the
hold analysis, move generation, and Rate My Beta all speak with the same
experienced-climber voice.
"""

GRIP_REFERENCE = """GRIP TYPES — how to hold what you find:
- JUG: wrap the whole hand, relaxed grip. Rest spots — shake out here.
- EDGE / CRIMP: three grips, escalating stress:
  * open hand / drag — fingers draped, least tendon stress, best for endurance; default on bigger edges
  * half crimp — fingers at 90 degrees, thumb beside fingers; strong and relatively safe; the workhorse grip on small edges
  * full crimp — thumb locked over the fingertips; maximum force on the smallest edges but highest injury risk; use only when the move demands it, never on pockets
- SLOPER: open palm, maximum skin contact, wrist below the hold. Slopers are body-position holds — keep hips low and close, load them from directly below. They get worse when you're pumped or it's warm.
- PINCH: thumb opposition is the point — actively squeeze; keep the elbow tucked.
- POCKET: two-finger (middle+ring is strongest pair) or mono. ALWAYS open hand or half crimp — never full crimp a pocket.
- UNDERCLING: palm up, pull toward you. Weak overhead, strongest at waist-to-chest height — get feet high FIRST, then the undercling lets you stand tall and reach far.
- SIDE PULL: vertical hold pulled sideways toward your midline. Lean away from it and oppose with feet — it only works with counterpressure.
- GASTON: like opening a sliding door — elbow out, pushing away from your midline. Shoulder-intensive; keep the shoulder engaged, not shrugged."""


_TECH_FUNDAMENTALS = """  Footwork: edge on the inside of the big toe | smear (flat shoe on wall, weight over the foot) | foot swap on small chips | quiet, precise feet
  Balance: inside flag / outside flag (free leg as counterweight — normal technique, not a last resort)
  Hands: match | straight arms between moves, hang the skeleton not the muscles
  Movement: static and controlled, hips close to the wall, rockover (weight fully over a high foot to stand up on it)"""

_TECH_INTERMEDIATE = """  Footwork: backstep (outside edge, hip turned into the wall — pairs with a same-side reach) | heel hook on ledges, aretes and volumes (pull with the hamstring like a third arm) | drop knee (rotate the inside knee down and in — locks the hip in, extends reach, takes weight off the arms)
  Hands: side pull and undercling positions with opposing feet
  Movement: deadpoint (controlled momentum, grab the hold at the apex where you are briefly weightless) | twist-lock (hip rotation to reach far with a bent, locked arm)"""

_TECH_ADVANCED = """  Footwork: toe hook (top of the foot pulls behind an arete/roof lip — keep the leg long and tense) | knee bar (jam thigh against one surface while the foot pushes an opposing hold — on a good knee bar you can let go with both hands) | bicycle (one foot pushes, the other toe-hooks the same or nearby hold — steep-roof body tension)
  Hands: gaston sequences, matched full crimps when the edge demands it
  Movement: dyno (fully airborne — sink the hips, drive from the legs, spot the target hold, catch at the apex) | mantle (press down and pivot over a ledge or top-out) | pogo/moment (swing a leg for momentum on big moves)"""

_ANGLE_NOTES = {
    "Slab (less than vertical)": "SLAB: this is a footwork and balance game. Weight over feet, smear liberally, tiny hand pressure. Dynamic moves are rare — precision beats power here.",
    "Vertical": "VERTICAL: balanced climbing. Straight arms, hips to the wall, flags for balance on every offset reach.",
    "Slight overhang": "SLIGHT OVERHANG: body tension starts to matter. Drop knees and backsteps keep hips close; expect to cut smaller footholds.",
    "Steep overhang": "STEEP OVERHANG: core tension is everything. Drop knees, heel hooks and toe hooks keep feet on; knee bars are gold when the geometry offers one. Straight arms or you will pump out.",
    "Roof": "ROOF: horizontal climbing. Heel hooks, toe hooks, bicycles and knee bars are not optional — they ARE the climbing. Keep maximum body tension; every cut foot costs you the send.",
}


def technique_library(difficulty, wall_angle):
    """Technique guidance for the move-generation prompt, scaled by climber
    level and wall angle."""
    if difficulty == "Beginner":
        techniques = ("TECHNIQUES (fundamentals — keep it simple and controlled):\n"
                      + _TECH_FUNDAMENTALS
                      + "\nAvoid heel hooks, toe hooks, drop knees and dynamic moves unless the hold layout leaves no alternative.")
    elif difficulty == "Intermediate":
        techniques = ("TECHNIQUES (solid repertoire — use technique where it clearly helps):\n"
                      + _TECH_FUNDAMENTALS + "\n" + _TECH_INTERMEDIATE
                      + "\nAdvanced moves (toe hook, knee bar, dyno) only when the geometry clearly calls for them.")
    else:
        techniques = ("TECHNIQUES (full arsenal — an experienced climber reads the wall and uses what fits):\n"
                      + _TECH_FUNDAMENTALS + "\n" + _TECH_INTERMEDIATE + "\n" + _TECH_ADVANCED
                      + "\nDrop knees, hooks and flags are expected tools — reach for them proactively when they improve position, not as last resorts.")

    angle = _ANGLE_NOTES.get(wall_angle, "")
    return f"{techniques}\n\n{angle}\n\n{GRIP_REFERENCE}"


def coaching_knowledge():
    """Injected into the Rate My Beta coach so its feedback reflects real
    climbing experience."""
    return (GRIP_REFERENCE + "\n\nTECHNIQUE VOCABULARY you can reference when coaching:\n"
            + _TECH_FUNDAMENTALS + "\n" + _TECH_INTERMEDIATE + "\n" + _TECH_ADVANCED)
