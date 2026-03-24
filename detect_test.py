from inference_sdk import InferenceHTTPClient
from dotenv import load_dotenv
import os

load_dotenv()

CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=os.getenv("ROBOFLOW_API_KEY")
)

result = CLIENT.infer("wall.jpg", model_id="climbing-detection-bj3mn/1")

for detection in result["predictions"]:
    print(f"Hold found: {detection['class']} at x={detection['x']:.0f}, y={detection['y']:.0f} confidence={detection['confidence']:.2f}")