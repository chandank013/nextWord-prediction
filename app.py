import streamlit as st
import numpy as np
import time
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Next Word Prediction System",
    page_icon="🧠",
    layout="centered"
)

# THEME TOGGLE
theme = st.sidebar.radio("🌗 Theme", ["Light", "Dark"])

if theme == "Dark":
    bg = "#0f172a"
    card = "#1e293b"
    card_light = "#243447"
    text = "#e5e7eb"
    accent = "#38bdf8"
    btn_text = "#000000"
else:
    bg = "#f8fafc"
    card = "#e2e8f0"
    card_light = "#edf2f7"
    text = "#020617"
    accent = "#2563eb"
    btn_text = "#ffffff"

# CSS 
st.markdown(
    f"""
    <style>
    .stApp {{
        background-color: {bg};
        color: {text};
    }}

    h1 {{
        text-align: center;
        font-weight: 800;
        color: {accent};
    }}

    .card {{
        background-color: {card};
        padding: 18px;
        border-radius: 14px;
        margin-top: 15px;
        font-size: 18px;
        box-shadow: 0 6px 16px rgba(0,0,0,0.15);
    }}

    .card-light {{
        background-color: {card_light};
        padding: 18px;
        border-radius: 14px;
        margin-top: 12px;
        font-size: 18px;
        box-shadow: 0 6px 16px rgba(0,0,0,0.12);
    }}

    .word {{
        font-weight: 700;
        color: {accent};
        font-size: 20px;
    }}

    button[kind="primary"] {{
        background: linear-gradient(90deg, {accent}, #22d3ee);
        color: {btn_text};
        font-weight: 700;
        border-radius: 12px;
        padding: 10px 22px;
        border: none;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# LOAD MODEL & TOKENIZER 
model = load_model("models/next_word_prediction_model.h5", compile=False)

with open("models/tokenizer.pickle", "rb") as f:
    tokenizer = pickle.load(f)
max_sequence_len = model.input_shape[1] + 1

# ---------------- FUNCTIONS ----------------
def top_k_predictions(model, tokenizer, text, max_len, k=3):
    seq = tokenizer.texts_to_sequences([text.lower()])[0]

    if len(seq) >= max_len:
        seq = seq[-(max_len - 1):]

    padded = pad_sequences([seq], maxlen=max_len - 1, padding="pre")
    preds = model.predict(padded, verbose=0)[0]

    top_indices = preds.argsort()[-k:][::-1]
    return [(tokenizer.index_word.get(i, ""), preds[i]) for i in top_indices if i != 0]


def generate_sentence(model, tokenizer, seed_text, max_len, steps=12):
    text = seed_text
    for _ in range(steps):
        next_word = top_k_predictions(model, tokenizer, text, max_len, k=1)
        if not next_word:
            break
        text += " " + next_word[0][0]
    return text


def typing_effect_card(text):
    placeholder = st.empty()
    typed = ""
    for char in text:
        typed += char
        placeholder.markdown(
            f"<div class='card-light'>{typed}</div>",
            unsafe_allow_html=True
        )
        time.sleep(0.025)

# ---------------- UI ----------------
st.title("🧠 Next Word Prediction System")
st.markdown("✍️ **Enter a sequence of words**")
user_input = st.text_input("", "Ham. Vpon what")

col1, col2 = st.columns(2)

with col1:
    predict_btn = st.button("🔮 Predict Next Words")
with col2:
    generate_btn = st.button("📝 Generate Sentence")

# TOP-3 PREDICTIONS
if predict_btn:
    if user_input.strip():
        predictions = top_k_predictions(
            model, tokenizer, user_input, max_sequence_len, k=3
        )

        if predictions:
            st.markdown(
                "<div class='card'><b>Top-3 Predicted Words:</b></div>",
                unsafe_allow_html=True
            )

            for word, prob in predictions:
                st.markdown(
                    f"<div class='card-light'>➡️ "
                    f"<span class='word'>{word}</span> "
                    f"(confidence: {prob:.2f})</div>",
                    unsafe_allow_html=True
                )
        else:
            st.error("Could not predict the next word.")
    else:
        st.warning("Please enter some text.")

# GENERATED SENTENCE CARD
if generate_btn:
    if user_input.strip():
        sentence = generate_sentence(
            model, tokenizer, user_input, max_sequence_len, steps=14
        )

        # Header card (like your screenshot)
        st.markdown(
            "<div class='card'><b>Generated Sentence:</b></div>",
            unsafe_allow_html=True
        )

        # Typing animation inside card
        typing_effect_card(sentence)

    else:
        st.warning("Please enter some text.")
