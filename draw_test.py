from inference_sdk import InferenceHTTPClient
from dotenv import load_dotenv
from PIL import Image, ImageDraw, ImageFont
import os

load_dotenv()

CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=os.getenv("ROBOFLOW_API_KEY")
)

result = CLIENT.infer("wall.jpg", model_id="climbing-detection-bj3mn/1")

image = Image.open("wall.jpg").convert("RGB")
draw = ImageDraw.Draw(image)

for i, detection in enumerate(result["predictions"]):
    x, y = detection["x"], detection["y"]
    w, h = detection["width"], detection["height"]
    
    # Draw a box around each hold
    draw.rectangle([x - w/2, y - h/2, x + w/2, y + h/2], outline="lime", width=3)
    
    # Number each hold
    draw.ellipse([x-15, y-15, x+15, y+15], fill="lime")
    draw.text((x-5, y-8), str(i+1), fill="black")

image.save("wall_detected.jpg")
print(f"Done! Detected {len(result['predictions'])} holds. Saved to wall_detected.jpg")