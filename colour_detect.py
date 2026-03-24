from inference_sdk import InferenceHTTPClient
from PIL import Image, ImageDraw
from dotenv import load_dotenv
import cv2
import numpy as np
import os

load_dotenv()

CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=os.getenv("ROBOFLOW_API_KEY")
)

# Colour ranges in HSV format (hue, saturation, value)
COLOUR_RANGES = {
    "Black":  [(0, 0, 0), (180, 80, 50)],
    "Blue":   [(90, 50, 50), (130, 255, 255)],
    "Red":    [(0, 100, 100), (10, 255, 255)],
    "Green":  [(40, 50, 50), (80, 255, 255)],
    "Orange": [(10, 100, 100), (25, 255, 255)],
    "Pink":   [(140, 50, 100), (170, 255, 255)],
    "White":  [(0, 0, 180), (180, 30, 255)],
    "Yellow": [(25, 100, 100), (35, 255, 255)],
    "Purple": [(130, 50, 50), (160, 255, 255)],
}

def is_colour(image_crop, colour_name):
    hsv = cv2.cvtColor(image_crop, cv2.COLOR_BGR2HSV)
    low, high = COLOUR_RANGES[colour_name]
    mask = cv2.inRange(hsv, np.array(low), np.array(high))
    match_percent = np.sum(mask > 0) / mask.size
    return match_percent > 0.15  # at least 15% of pixels match

def detect_holds(image_path, colour):
    # Run detection
    result = CLIENT.infer(image_path, model_id="climbing-detection-bj3mn/1")
    
    # Load image with OpenCV for colour checking
    cv_image = cv2.imread(image_path)
    
    # Load image with PIL for drawing
    pil_image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(pil_image)
    
    matched_holds = []
    hold_number = 1
    
    for detection in result["predictions"]:
        x, y = int(detection["x"]), int(detection["y"])
        w, h = int(detection["width"]), int(detection["height"])
        
        # Crop the hold area from the image
        x1, y1 = max(0, x - w//2), max(0, y - h//2)
        x2, y2 = min(cv_image.shape[1], x + w//2), min(cv_image.shape[0], y + h//2)
        crop = cv_image[y1:y2, x1:x2]
        
        if crop.size == 0:
            continue
            
        # Check if this hold matches the selected colour
        if is_colour(crop, colour):
            matched_holds.append({"number": hold_number, "x": x, "y": y})
            
            # Draw box and number on matched holds only
            draw.rectangle([x1, y1, x2, y2], outline="lime", width=3)
            draw.ellipse([x-15, y-15, x+15, y+15], fill="lime")
            draw.text((x-5, y-8), str(hold_number), fill="black")
            hold_number += 1
    
    pil_image.save("wall_colour_detected.jpg")
    print(f"Found {len(matched_holds)} {colour} holds")
    return matched_holds

# Test it
holds = detect_holds("wall.jpg", "Black")
for h in holds:
    print(f"Hold {h['number']}: x={h['x']}, y={h['y']}")