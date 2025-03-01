import streamlit as st
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional

# Streamlit UI Styling
st.set_page_config(page_title="Roman Urdu Poetry Generator", page_icon="🎵", layout="centered")

st.markdown(
    """
    <style>
    .title {
        text-align: center;
        color: #E63946;
        font-size: 36px;
        font-weight: bold;
    }
    .generated-text {
        background-color: #F1FAEE;
        padding: 15px;
        border-radius: 10px;
        color: #1D3557;
        font-size: 20px;
        text-align: center;
        font-family: cursive;
    }
    .stButton>button {
        background-color: #457B9D;
        color: white;
        font-size: 18px;
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<h1 class="title">Roman Urdu Poetry Generator 🎤</h1>', unsafe_allow_html=True)

# Load Tokenizer & Prepare Data
tokenizer = Tokenizer()
data = open('romanized_urdu_poetry_3.txt').read()
corpus = data.lower().split("\n")
tokenizer.fit_on_texts(corpus)
total_words = len(tokenizer.word_index) + 1

# Determine max_sequence_len from training data
max_sequence_len = max(len(tokenizer.texts_to_sequences([line])[0]) for line in corpus)

# Rebuild the model architecture
model = Sequential([
    Embedding(total_words, 100, input_length=max_sequence_len - 1),
    Bidirectional(LSTM(150, return_sequences=True)),
    Dropout(0.2),
    LSTM(100),
    Dense(total_words, activation='softmax')
])

# Build the model before loading weights
model.build(input_shape=(None, max_sequence_len - 1))

# Load saved weights
model.load_weights("/content/model.weights.h5")

# Poetry Generation Function
def generate_poetry(seed_text, words_per_line=6, total_lines=4):
    poem = ""
    for _ in range(total_lines):
        current_line = seed_text
        generated_line = ""

        for _ in range(words_per_line):
            token_list = tokenizer.texts_to_sequences([current_line])[0]
            token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')

            predicted = np.argmax(model.predict(token_list, verbose=0), axis=-1)
            output_word = next((word for word, index in tokenizer.word_index.items() if index == predicted), "")
            generated_line += output_word + " "
            current_line += " " + output_word  

        poem += generated_line.strip() + "\n"
        seed_text = " ".join(generated_line.split()[-3:])  # Use last 3 words as the new seed

    return poem

# User Input Box
seed_text = st.text_input("Enter a seed phrase for poetry:", value="kha ho tum")

# Generate Poetry Button
if st.button("Generate Poetry 🎶"):
    poetry = generate_poetry(seed_text)
    st.markdown('<h3 class="generated-text">{}</h3>'.format(poetry.replace("\n", "<br>")), unsafe_allow_html=True)

st.markdown("### 🎨 **Enjoy the magic of AI poetry!**")
