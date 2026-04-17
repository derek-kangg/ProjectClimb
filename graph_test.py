from route_graph import build_reachability_graph, format_graph_for_prompt
from PIL import Image

# Use the holds we know from our wall.jpg
holds = [
    {"number": 1, "x": 470, "y": 932, "size": "small (likely foothold)"},
    {"number": 2, "x": 186, "y": 933, "size": "large (likely handhold)"},
    {"number": 3, "x": 434, "y": 751, "size": "small (likely foothold)"},
    {"number": 4, "x": 717, "y": 681, "size": "large (likely handhold)"},
    {"number": 5, "x": 259, "y": 660, "size": "medium (likely handhold)"},
    {"number": 6, "x": 371, "y": 510, "size": "medium (likely handhold)"},
    {"number": 7, "x": 494, "y": 422, "size": "medium (likely handhold)"},
    {"number": 8, "x": 620, "y": 243, "size": "large (likely handhold)"},
    {"number": 9, "x": 484, "y": 89,  "size": "large (likely handhold)"},
]

image = Image.open("wall.jpg")
image_width, image_height = image.size

graph = build_reachability_graph(holds, image_height, image_width, climber_height_cm=182)
output = format_graph_for_prompt(graph, holds, start_hold_numbers=[5])

print(output)