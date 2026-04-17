from PIL import Image, ImageDraw
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

def detect_holds_by_colour(image_path, colour, min_area=70):
    cv_image = cv2.imread(image_path)
    hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

    low, high = COLOUR_RANGES[colour]
    mask = cv2.inRange(hsv, np.array(low), np.array(high))

    # Clean up the mask
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Find contours
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

    pil_image.save("blob_test.jpg")
    print(f"Found {len(holds)} {colour} holds")
    for h in holds:
        print(f"Hold {h['number']}: x={h['x']}, y={h['y']}")

    return holds

detect_holds_by_colour("wall.jpg", "Black")