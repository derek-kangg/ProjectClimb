from ultralytics import YOLO
from PIL import Image, ImageDraw
import cv2

# This downloads a small pre-trained model automatically on first run
model = YOLO("yolov8n.pt")

results = model("wall.jpg")

image = Image.open("wall.jpg").convert("RGB")
draw = ImageDraw.Draw(image)

count = 0
for result in results:
    for box in result.boxes:
        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
        conf = float(box.conf[0])
        if conf > 0.3:
            draw.rectangle([x1, y1, x2, y2], outline="lime", width=3)
            count += 1

image.save("yolo_test.jpg")
print(f"Detected {count} objects")