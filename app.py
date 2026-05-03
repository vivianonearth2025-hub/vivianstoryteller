# Storytelling App for kids (3-10 years old)

# Import part
import streamlit as st
from transformers import pipeline


# Function part

# Cache the models so we don't load them every time
@st.cache_resource
def load_image_to_text_model():
    return pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")


@st.cache_resource
def load_story_generator_model():
    return pipeline("text-generation", model="gpt2")


@st.cache_resource
def load_text_to_audio_model():
    return pipeline("text-to-audio", model="Matthijs/mms-tts-eng")


# img2text
def img2text(url):
    image_to_text_model = load_image_to_text_model()
    text = image_to_text_model(url)[0]["generated_text"]
    return text


# Words we don't want in a kids story
UNSAFE_WORDS = [
    "kill", "killed", "killing", "murder", "murdered", "death", "die", "died",
    "dying", "dead", "blood", "bloody", "gun", "shoot", "shot", "weapon",
    "knife", "stab", "beat", "beaten", "beating", "fight", "fought", "war",
    "attack", "attacked", "hit", "hurt", "hurts",
    "rape", "raped", "abuse", "abused", "sex", "sexual", "naked", "nude",
    "marriage", "married", "wife", "husband", "boyfriend", "girlfriend",
    "kiss", "kissed", "love", "lover",
    "drug", "drugs", "drunk", "alcohol", "beer", "wine", "smoke", "cigarette",
    "suicide", "asylum", "prison", "jail", "police", "cheated", "cheat",
    "steal", "stolen", "stealing", "rob", "robbed", "scary", "scared",
    "monster", "demon", "devil", "ghost",
    "hate", "hated", "angry", "sad", "cry", "cried", "crying", "lonely",
    "hell", "damn",
]


def is_kid_friendly(story):
    story_lower = story.lower()
    for word in UNSAFE_WORDS:
        if (f" {word} " in f" {story_lower} "
                or f" {word}." in story_lower
                or f" {word}," in story_lower
                or f" {word}!" in story_lower
                or f" {word}?" in story_lower):
            return False
    if len(story.split()) < 30:
        return False
    return True


def clean_story(text):
    if not text:
        return ""

    # Remove leading symbols
    while text and not text[0].isalpha():
        text = text[1:]

    if not text:
        return ""

    # Skip bad first-person openings
    bad_starts = ["i'm ", "i am ", "i ", "you ", "we ", "they "]
    text_lower = text.lower()
    for bs in bad_starts:
        if text_lower.startswith(bs):
            return ""
    return text.strip()


# text2story
def text2story(text):
    # Prompt for the story
    prompt = (
        f"Here is a happy and gentle children's story about {text}.\n\n"
        f"Once upon a sunny day, {text}. They were having so much fun together. "
    )

    story_pipe = load_story_generator_model()

    story_text = ""
    for attempt in range(5):
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

        # Remove the prompt from the result
        candidate = full_output.replace(prompt, "").strip()

        candidate = clean_story(candidate)
        if not candidate:
            continue

        # Keep at most 100 words and end on a full sentence
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

        if is_kid_friendly(candidate):
            story_text = candidate
            break
        story_text = candidate

    # Backup story if all attempts fail
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


# text2audio
def text2audio(story_text):
    audio_pipe = load_text_to_audio_model()
    audio_data = audio_pipe(story_text)
    return audio_data


# Main part
def main():
    st.set_page_config(page_title="Magic Story Maker", page_icon="🦄")

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
    st.write("")

    uploaded_file = st.file_uploader(
        "Choose an image to start your story adventure! 🌈",
        type=["png", "jpg", "jpeg"],
    )

    if uploaded_file is not None:
        # Save file locally
        bytes_data = uploaded_file.getvalue()
        with open(uploaded_file.name, "wb") as file:
            file.write(bytes_data)

        st.image(uploaded_file, caption="Your Picture 🖼️", use_column_width=True)

        file_key = uploaded_file.name

        # Only run pipeline when a new image is uploaded
        if st.session_state.get("file_key") != file_key:
            # Stage 1
            with st.spinner("Looking at your picture... 👀"):
                scenario = img2text(file_key)

            # Stage 2
            with st.spinner("Writing your story... ✏️"):
                story = text2story(scenario)

            # Stage 3
            with st.spinner("Getting ready to read it out loud... 🎤"):
                audio_data = text2audio(story)

            st.session_state["file_key"] = file_key
            st.session_state["scenario"] = scenario
            st.session_state["story"] = story
            st.session_state["audio_data"] = audio_data

        scenario = st.session_state["scenario"]
        story = st.session_state["story"]
        audio_data = st.session_state["audio_data"]

        # Show caption
        st.success("I can see what's in your picture! 🎉")
        with st.expander("🔍 What I see in the picture"):
            st.write(scenario)

        # Show story
        st.success("Your story is ready! 📖")
        st.markdown("### 📖 Your Magical Story")
        st.markdown(
            f"<div style='background-color: #FFF8DC; padding: 20px; "
            f"border-radius: 15px; font-size: 20px; line-height: 1.6; "
            f"color: #333;'>{story}</div>",
            unsafe_allow_html=True,
        )
        st.write("")

        # Play audio
        st.markdown("### 🔊 Listen to Your Story!")
        if st.button("▶️ Play My Story!"):
            audio_array = audio_data["audio"]
            sample_rate = audio_data["sampling_rate"]
            st.audio(audio_array, sample_rate=sample_rate)
            st.balloons()


if __name__ == "__main__":
    main()
