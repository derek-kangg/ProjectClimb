import cv2
import numpy as np
from PIL import Image, ImageDraw

def detect_bolt_holes(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Focus on the white wall area by masking dark regions
    _, white_mask = cv2.threshold(gray, 170, 255, cv2.THRESH_BINARY)
    
    # Blur slightly for circle detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 1)
    
    # Detect circles with much smaller radius range
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=15,        # bolt holes are evenly spaced
        param1=30,         # lower edge threshold
        param2=10,         # lower detection threshold to catch more circles
        minRadius=2,       # bolt holes are very small
        maxRadius=7        # bolt holes are very small
    )
    
    if circles is None:
        print("No bolt holes detected")
        return None, None
    
    circles = np.round(circles[0, :]).astype("int")
    
    # Filter to only keep circles that are dark (bolt holes are dark dots)
    filtered = []
    for (x, y, r) in circles:
        # Sample the pixel brightness at the circle center
        brightness = gray[y, x]
        # Check the surrounding white wall area
        surrounding = gray[max(0,y-10):y+10, max(0,x-10):x+10]
        avg_surrounding = np.mean(surrounding)
        
        # Bolt hole should be dark relative to surrounding wall
        if brightness < 100 and avg_surrounding > 150:
            filtered.append((x, y, r))
    
    print(f"Detected {len(filtered)} bolt holes after filtering")
    
    # Draw detected bolt holes on image
    pil_image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(pil_image)
    
    for (x, y, r) in filtered:
        draw.ellipse([x-r-2, y-r-2, x+r+2, y+r+2], outline="red", width=2)
    
    pil_image.save("bolt_detected.jpg")
    
    if len(filtered) < 3:
        print("Not enough bolt holes detected for calibration")
        return filtered, None
    
    # Calculate average distance between nearby bolt holes
    distances = []
    for i, (x1, y1, r1) in enumerate(filtered):
        nearest_dists = []
        for j, (x2, y2, r2) in enumerate(filtered):
            if i == j:
                continue
            dist = ((x2-x1)**2 + (y2-y1)**2) ** 0.5
            nearest_dists.append(dist)
        
        if nearest_dists:
            nearest_dists.sort()
            distances.append(nearest_dists[0])
    
    if distances:
        avg_bolt_distance_px = np.median(distances)
        BOLT_SPACING_CM = 20
        pixels_per_cm = avg_bolt_distance_px / BOLT_SPACING_CM
        print(f"Average bolt hole spacing: {avg_bolt_distance_px:.1f} pixels")
        print(f"Estimated scale: {pixels_per_cm:.2f} pixels per cm")
        return filtered, pixels_per_cm
    
    return filtered, None

circles, pixels_per_cm = detect_bolt_holes("wall.jpg")
print(f"\nFinal scale estimate: {pixels_per_cm:.2f} pixels/cm" if pixels_per_cm else "Could not calculate scale")