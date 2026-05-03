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
    """Load the story generation model (cached).
    Using GPT-2 because it follows the prompt more faithfully than
    specialized story models, which often ignore the input and recite
    memorized story fragments from their training data.
    """
    return pipeline("text-generation", model="gpt2")


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
    # Violence
    "kill", "killed", "killing", "murder", "murdered", "death", "die", "died",
    "dying", "dead", "blood", "bloody", "gun", "shoot", "shot", "weapon",
    "knife", "stab", "beat", "beaten", "beating", "fight", "fought", "war",
    "attack", "attacked", "hit", "hurt", "hurts",
    # Adult content
    "rape", "raped", "abuse", "abused", "sex", "sexual", "naked", "nude",
    "marriage", "married", "wife", "husband", "boyfriend", "girlfriend",
    "kiss", "kissed", "love", "lover",
    # Substances
    "drug", "drugs", "drunk", "alcohol", "beer", "wine", "smoke", "cigarette",
    # Crime / scary
    "suicide", "asylum", "prison", "jail", "police", "cheated", "cheat",
    "steal", "stolen", "stealing", "rob", "robbed", "scary", "scared",
    "monster", "demon", "devil", "ghost",
    # Negative emotions
    "hate", "hated", "angry", "sad", "cry", "cried", "crying", "lonely",
    # Profanity
    "hell", "damn",
]


def is_kid_friendly(story):
    """Check if the story is appropriate for 3-10 year old kids."""
    story_lower = story.lower()
    for word in UNSAFE_WORDS:
        # Use spaces to match whole words only (avoid false positives like
        # "diet" matching "die")
        if (f" {word} " in f" {story_lower} "
                or f" {word}." in story_lower
                or f" {word}," in story_lower
                or f" {word}!" in story_lower
                or f" {word}?" in story_lower):
            return False
    # Reject if too short
    if len(story.split()) < 30:
        return False
    return True


def clean_story(text):
    """Clean up messy characters that GPT-2 sometimes produces at the start."""
    while text and not text[0].isalpha():
        text = text[1:]
        bad_starts = ["i'm ", "i am ", "i ", "you ", "we ", "they "]
    text_lower = text.lower()
    for bs in bad_starts:
        if text_lower.startswith(bs):
            return ""  # signal to regenerate
    return text.strip()


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
    # Build a prompt that anchors GPT-2 firmly to the image content.
    # Strategy: state the topic, then START the story so GPT-2 just continues it.
    prompt = (
        f"Here is a happy and gentle children's story about {text}.\n\n"
        f"Once upon a sunny day, {text}. They were having so much fun together. "
    )

    # Load the cached story generator
    story_pipe = load_story_generator_model()

    # Try up to 5 times to get a kid-friendly, well-formed story
    story_text = ""
    for attempt in range(5):
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

        # Clean up messy openings (~~~, "I'm talking to...", etc.)
        candidate = clean_story(candidate)
        if not candidate:
            continue  # try again

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

    # Final safety net: if after 5 tries the story is still not clean,
    # fall back to a fixed safe story template based on the caption
    if not story_text or not is_kid_friendly(story_text):
        story_text = (
            f"Once upon a sunny day, {text}. They were having so much fun "
            f"together. The sky was bright blue and the birds were singing "
            f"sweet songs. Everyone laughed and played all afternoon, sharing "
            f"snacks and telling silly jokes. When the sun started to set, "
            f"they all walked home with happy hearts, looking forward to "
            f"another wonderful day tomorrow. The end."
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

        # Use the file name as a key so the story regenerates only when the
        # user uploads a new image (not on every button click)
        file_key = uploaded_file.name

        # If this is a new image OR we haven't generated yet, run the pipeline.
        # Otherwise, reuse the cached results from session_state.
        if st.session_state.get("file_key") != file_key:
            # ---- Stage 1: Image to Text ----
            with st.spinner("Looking at your picture... 👀"):
                scenario = img2text(file_key)

            # ---- Stage 2: Text to Story ----
            with st.spinner("Writing your story... ✏️"):
                story = text2story(scenario)

            # ---- Stage 3: Text to Audio ----
            with st.spinner("Getting ready to read it out loud... 🎤"):
                audio_data = text2audio(story)

            # Save everything to session_state so we don't regenerate on
            # every interaction (like clicking the Play button)
            st.session_state["file_key"] = file_key
            st.session_state["scenario"] = scenario
            st.session_state["story"] = story
            st.session_state["audio_data"] = audio_data

        # Read the (possibly cached) results from session_state
        scenario = st.session_state["scenario"]
        story = st.session_state["story"]
        audio_data = st.session_state["audio_data"]

        # ---- Display: Caption ----
        st.success("I can see what's in your picture! 🎉")
        with st.expander("🔍 What I see in the picture"):
            st.write(scenario)

        # ---- Display: Story ----
        st.success("Your story is ready! 📖")
        st.markdown("### 📖 Your Magical Story")
        st.markdown(
            f"<div style='background-color: #FFF8DC; padding: 20px; "
            f"border-radius: 15px; font-size: 20px; line-height: 1.6; "
            f"color: #333;'>{story}</div>",
            unsafe_allow_html=True,
        )
        st.write("")  # spacer

        # ---- Display: Audio playback ----
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
