# Program title: Storytelling App for Kids
# Description: An image-to-story-to-audio application designed for 3-10 year old kids.
#              Users upload an image, the app generates a kid-friendly story, and reads it aloud.

# Import part
import streamlit as st
from transformers import pipeline


# ============================================================
# Function part
# ============================================================

# ---------- Cached model loaders ----------
# Use @st.cache_resource to load each model only once,
# so the app stays fast even when users upload multiple images.
@st.cache_resource
def load_image_to_text_model():
    """Load the BLIP image captioning model (cached)."""
    return pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")


@st.cache_resource
def load_story_generator_model():
    """Load the story generation model (cached)."""
    return pipeline("text-generation", model="pranavpsv/genre-story-generator-v2")


@st.cache_resource
def load_text_to_audio_model():
    """Load the text-to-speech model (cached)."""
    return pipeline("text-to-audio", model="Matthijs/mms-tts-eng")


# ---------- Function 1: Image to Text ----------
def img2text(url):
    """
    Stage 1: Extract a caption (scenario) from the uploaded image.
    Args:
        url (str): Path to the image file.
    Returns:
        str: A short text description of the image.
    """
    image_to_text_model = load_image_to_text_model()
    text = image_to_text_model(url)[0]["generated_text"]
    return text


# ---------- Function 2: Text to Story ----------
# Words that are not appropriate for a 3-10 year old audience.
# If the generated story contains any of these, we will regenerate it.
UNSAFE_WORDS = [
    "kill", "killed", "killing", "murder", "murdered", "death", "die", "died",
    "dying", "blood", "bloody", "gun", "shoot", "shot", "weapon", "knife",
    "stab", "beat", "beaten", "beating", "rape", "raped", "abuse", "abused",
    "drug", "drugs", "drunk", "alcohol", "sex", "sexual", "naked", "nude",
    "hell", "damn", "hate", "hated", "suicide", "asylum", "prison", "jail",
    "police", "cheated", "cheat", "steal", "stolen",
]


def is_kid_friendly(story):
    """Check if the story is appropriate for 3-10 year old kids."""
    story_lower = story.lower()
    for word in UNSAFE_WORDS:
        # Use spaces to match whole words only (avoid false positives like
        # "diet" matching "die")
        if f" {word} " in f" {story_lower} " or f" {word}." in story_lower \
                or f" {word}," in story_lower:
            return False
    return True


def text2story(text):
    """
    Stage 2: Turn the image caption into a kid-friendly story (50-100 words).
    A storytelling-style prompt is used to guide the model into a fairy-tale
    voice, and the result is filtered to make sure it is appropriate for kids.
    Args:
        text (str): The caption from img2text().
    Returns:
        str: A kid-friendly story.
    """
    # Build a fairy-tale style prompt. Starting with "Once upon a time" tells
    # the model "this is a children's bedtime story", which works much better
    # than a meta-instruction like "Write a story for kids".
    prompt = (
        f"Once upon a time, in a happy and magical world, there was "
        f"{text}. This is a sweet bedtime story for little children. "
    )

    # Load the cached story generator
    story_pipe = load_story_generator_model()

    # Try up to 3 times to get a kid-friendly story
    story_text = ""
    for attempt in range(3):
        # Generate with controlled length and safer sampling parameters.
        # Lower temperature => more focused, less random output.
        # repetition_penalty => avoid the model repeating the same phrases.
        story_results = story_pipe(
            prompt,
            max_new_tokens=140,
            min_new_tokens=70,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.2,
            truncation=True,
            pad_token_id=50256,
        )
        full_output = story_results[0]["generated_text"]

        # Strip the prompt from the output so only the story remains
        candidate = full_output.replace(prompt, "").strip()

        # Trim to <=100 words while keeping the last sentence complete
        words = candidate.split()
        if len(words) > 100:
            truncated = " ".join(words[:100])
            last_end = max(
                truncated.rfind("."),
                truncated.rfind("!"),
                truncated.rfind("?"),
            )
            if last_end > 0:
                candidate = truncated[: last_end + 1]
            else:
                candidate = truncated + "."

        # Check if the candidate is safe for kids; if so, use it.
        # Otherwise, try again.
        if is_kid_friendly(candidate):
            story_text = candidate
            break
        story_text = candidate  # keep the latest in case all attempts fail

    # Final safety net: if after 3 tries the story is still not clean,
    # prepend a friendly opener so it at least feels like a children's story
    if not is_kid_friendly(story_text):
        story_text = (
            f"Once upon a time, there was {text}. They had a wonderful "
            f"sunny day full of laughter, friends, and fun adventures. "
            f"Everyone smiled and played happily together until it was "
            f"time to go home. The end."
        )

    return story_text


# ---------- Function 3: Text to Audio ----------
def text2audio(story_text):
    """
    Stage 3: Convert the story into speech.
    Args:
        story_text (str): The story generated by text2story().
    Returns:
        dict: A dict with "audio" (numpy array) and "sampling_rate" (int).
    """
    audio_pipe = load_text_to_audio_model()
    audio_data = audio_pipe(story_text)
    return audio_data


# ---------- Function 4: Main ----------
def main():
    """Main function: builds the kid-friendly Streamlit UI and runs the pipeline."""

    # ---- Page config (kid-friendly title and icon) ----
    st.set_page_config(page_title="Magic Story Maker", page_icon="🦄")

    # ---- Header ----
    st.markdown(
        "<h1 style='text-align: center; color: #FF6F91;'>🦄 Magic Story Maker! ✨</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<h4 style='text-align: center; color: #6A89CC;'>"
        "📸 Pick a picture and I'll tell you a story! 📖"
        "</h4>",
        unsafe_allow_html=True,
    )
    st.write("")  # spacer

    # ---- File uploader ----
    uploaded_file = st.file_uploader(
        "Choose an image to start your story adventure! 🌈",
        type=["png", "jpg", "jpeg"],
    )

    if uploaded_file is not None:
        # Save uploaded file locally so the pipeline can read it
        bytes_data = uploaded_file.getvalue()
        with open(uploaded_file.name, "wb") as file:
            file.write(bytes_data)

        # Show the uploaded image
        st.image(uploaded_file, caption="Your Picture 🖼️", use_column_width=True)

        # ---- Stage 1: Image to Text ----
        with st.spinner("Looking at your picture... 👀"):
            scenario = img2text(uploaded_file.name)
        st.success("I can see what's in your picture! 🎉")
        with st.expander("🔍 What I see in the picture"):
            st.write(scenario)

        # ---- Stage 2: Text to Story ----
        with st.spinner("Writing your story... ✏️"):
            story = text2story(scenario)
        st.success("Your story is ready! 📖")
        st.markdown("### 📖 Your Magical Story")
        st.markdown(
            f"<div style='background-color: #FFF8DC; padding: 20px; "
            f"border-radius: 15px; font-size: 20px; line-height: 1.6; "
            f"color: #333;'>{story}</div>",
            unsafe_allow_html=True,
        )
        st.write("")  # spacer

        # ---- Stage 3: Text to Audio ----
        with st.spinner("Getting ready to read it out loud... 🎤"):
            audio_data = text2audio(story)

        # ---- Audio playback ----
        st.markdown("### 🔊 Listen to Your Story!")
        if st.button("▶️ Play My Story!"):
            audio_array = audio_data["audio"]
            sample_rate = audio_data["sampling_rate"]
            st.audio(audio_array, sample_rate=sample_rate)
            st.balloons()  # 🎈 fun touch for kids


# ============================================================
# Run the app
# ============================================================
if __name__ == "__main__":
    main()
