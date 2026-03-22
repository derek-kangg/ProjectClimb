import streamlit as st
from openai import OpenAI
import base64

client = OpenAI(api_key="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")

st.title("🧗 ProjectClimb")
st.write("Upload a photo of a climbing wall and get a route breakdown!")

uploaded_file = st.file_uploader("Upload a climbing wall photo", type=["jpg", "jpeg", "png"])

colour = st.selectbox("Which colour route do you want to climb?", 
    ["Black", "Blue", "Red", "Green", "Orange", "Pink", "White", "Yellow", "Purple"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Your uploaded wall", use_column_width=True)
    
    if st.button("Generate Route"):
        with st.spinner("Analysing your route..."):
            image_data = base64.b64encode(uploaded_file.read()).decode("utf-8")
            
            message = client.chat.completions.create(
                model="gpt-4o",
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}
                        },
                        {
                            "type": "text",
                            "text": f"This is a photo of a rock climbing wall. Focus ONLY on the {colour} holds. Ignore all other colours. Suggest a beginner-friendly route using only the {colour} holds, explaining each move step by step."
                        }
                    ]
                }]
            )
            
            st.markdown(message.choices[0].message.content)