import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
import os
import base64

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

st.set_page_config(
    page_title="ProjectClimb",
    page_icon="🧗",
    layout="centered"
)

st.markdown("""
    <h1 style='text-align: center; color: #6fcf4a; font-size: 3rem;'>🧗 ProjectClimb</h1>
    <p style='text-align: center; color: #6b806b; margin-bottom: 2rem;'>
        Upload a photo of a climbing wall and get an expert route breakdown
    </p>
""", unsafe_allow_html=True)

st.divider()

col1, col2 = st.columns([2, 1])

with col1:
    uploaded_file = st.file_uploader("Upload a climbing wall photo", type=["jpg", "jpeg", "png"])

with col2:
    colour = st.selectbox("Route colour", 
        ["Black", "Blue", "Red", "Green", "Orange", "Pink", "White", "Yellow", "Purple"])
    
    difficulty = st.selectbox("Your experience level",
        ["Beginner", "Intermediate", "Advanced"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded wall", use_column_width=True)
    
    st.divider()
    
    if st.button("🔍 Analyse Route", use_container_width=True):
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
                            "text": f"""You are an expert rock climbing coach analyzing a climbing wall photo.
                            Focus ONLY on the {colour} holds. Ignore all other colours completely.
                            Tailor your advice for a {difficulty} climber.

                            Provide a detailed route breakdown with the following structure:

                            1. **Route Overview:** Briefly describe the overall route and its difficulty.

                            2. **Starting Position:** Describe exactly where to place both hands and both feet to begin.

                            3. **Step by Step Moves:** For each move describe:
                                - Which hand or foot to move and where
                                - Body positioning and weight distribution
                                - The climbing technique being used. Always explain the technique in simple terms in brackets after naming it. For example: "Use a heel hook (place your heel on top of a hold and use it to pull your body up)"
                                - A difficulty rating for that move (Easy / Medium / Hard)

                            4. **Key Tips:** 2-3 specific tips for completing this route successfully.

                            Keep the advice practical and specific to what you can see in the image."""
                        }
                    ]
                }]
            )
            
            st.markdown("### 📋 Route Breakdown")
            st.markdown(message.choices[0].message.content)
            
            st.divider()
            st.caption("ProjectClimb — helping climbers of all levels get through plateaus 🧗")