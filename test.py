from openai import OpenAI
from dotenv import load_dotenv
import os
import base64

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load and encode the image
with open("wall.jpg", "rb") as f:
    image_data = base64.b64encode(f.read()).decode("utf-8")

message = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}
                },
                {
                    "type": "text",
                    "text": "This is a photo of a rock climbing wall. Focus ONLY on the black holds. Ignore all other colours. Suggest a beginner-friendly route using only the black holds, explaining each move step by step."
                }
            ]
        }
    ]
)

print(message.choices[0].message.content)