import io
import json
import time

import streamlit as st
from PIL import Image
from google import genai
from google.genai import types


# --------------------------------------------------
# Configuration
# --------------------------------------------------

MODEL = "gemini-flash-lite-latest"

SYSTEM = """
You are an agriculture assistant for small farmers in India.

The user uploads a photo of a crop leaf or plant.

Reply ONLY with valid JSON in this exact shape:
{
  "crop_guess": string,
  "what_i_see": string,
  "likely_problems": [list of up to 3 short strings],
  "confidence": "LOW" or "MEDIUM" or "HIGH",
  "first_steps": [list of simple, low-cost actions],
  "see_an_expert_if": string
}

Rules:
- Never give pesticide names or doses.
- If the image is not a plant, say so in what_i_see and set confidence to LOW.
- Use simple words.
- Write all text values in the language the user asks for.
- Recommend consulting a local agriculture officer or Krishi Vigyan Kendra when expert help is needed.
"""

client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])


# --------------------------------------------------
# Prepare uploaded image
# --------------------------------------------------

def prepare_image(uploaded):
    img = Image.open(uploaded).convert("RGB")

    # Reduce large images before sending them to the model
    img.thumbnail((1024, 1024))

    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=85)

    return buffer.getvalue()


# --------------------------------------------------
# Ask Gemini to analyze the plant
# --------------------------------------------------

def diagnose(image_bytes, crop, language, retries=3):

    image_part = types.Part.from_bytes(
        data=image_bytes,
        mime_type="image/jpeg"
    )

    note = f"Crop: {crop or 'unknown'}. Reply in {language}."

    for attempt in range(retries):

        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=[image_part, note],
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM,
                    response_mime_type="application/json",
                ),
            )

            return json.loads(response.text)

        except Exception as error:

            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                return {
                    "error": str(error)
                }


# --------------------------------------------------
# Streamlit page
# --------------------------------------------------

st.set_page_config(
    page_title="KisanMitra",
    page_icon="🌾",
    layout="centered"
)

st.title("🌾 KisanMitra")

st.caption(
    "Upload a crop or leaf photo for first-step guidance. "
    "This is not a replacement for an agriculture expert."
)


# --------------------------------------------------
# User inputs
# --------------------------------------------------

language = st.selectbox(
    "Reply language",
    ["English", "Tamil"]
)

crop = st.text_input(
    "Crop name (optional)",
    placeholder="For example: paddy, tomato, banana"
)

uploaded = st.file_uploader(
    "Upload a leaf or crop photo",
    type=["jpg", "jpeg", "png"]
)


# --------------------------------------------------
# Show uploaded image
# --------------------------------------------------

if uploaded:

    st.image(
        uploaded,
        caption="Your uploaded photo",
        use_container_width=True
    )


# --------------------------------------------------
# Analyze button
# --------------------------------------------------

if uploaded and st.button("🔍 Check my plant"):

    with st.spinner("Looking at the plant..."):

        image_bytes = prepare_image(uploaded)

        result = diagnose(
            image_bytes,
            crop,
            language
        )


    # --------------------------------------------------
    # Error handling
    # --------------------------------------------------

    if "error" in result:

        st.error(
            "Could not check the image right now. "
            "Please try again in a minute."
        )

    else:

        # Crop
        st.subheader(
            f"🌾 Crop: {result['crop_guess']}"
        )

        # What AI sees
        st.write(
            "**What I see:**"
        )

        st.write(
            result["what_i_see"]
        )


        # Confidence
        st.write(
            f"**Confidence:** {result['confidence']}"
        )


        # Possible problems
        st.write(
            "**⚠️ Possible problems:**"
        )

        for problem in result["likely_problems"]:

            st.write(
                f"- {problem}"
            )


        # First steps
        st.write(
            "**🌱 First steps you can try:**"
        )

        for step in result["first_steps"]:

            st.write(
                f"- {step}"
            )


        # Expert warning
        st.warning(
            "👨‍🌾 See an expert if: "
            + result["see_an_expert_if"]
        )


        # Safety message
        st.info(
            "Before spraying anything, show the photo to "
            "your local agriculture officer or nearest "
            "Krishi Vigyan Kendra (KVK)."
        )