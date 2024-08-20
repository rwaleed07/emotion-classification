import streamlit as st
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences



# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize tools
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Load the model
model_path = 'emotion_classification_model.h5'  # Replace with the actual path
model = load_model(model_path)


# Function to preprocess text
def preprocess_text(text):
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word.isalpha() and word not in stop_words]
    return " ".join(tokens)

# Streamlit app
st.title("Emotion Classification App")

st.write("This app classifies emotions from text input.")

# Text input from the user
user_input = st.text_area("Enter a sentence to classify:")

if st.button("Classify"):
    if user_input:
        # Preprocess the input
        processed_text = preprocess_text(user_input)

        # Convert text to sequence
        text_seq = tokenizer.texts_to_sequences([processed_text])
        max_len = 100  # Adjust based on training
        text_pad = pad_sequences(text_seq, maxlen=max_len)

        # Predict the emotion
        prediction = model.predict(text_pad)
        predicted_class = np.argmax(prediction, axis=1)[0]

        # Define emotion labels (adjust based on your dataset)
        emotion_labels = ['anger', 'love', 'fear', 'joy', 'sadness', 'surprise']  # Update according to your dataset
        predicted_emotion = emotion_labels[predicted_class]

        # Display the result
        st.write(f"The predicted emotion is: **{predicted_emotion.capitalize()}**")
    else:
        st.write("Please enter some text for classification.")
