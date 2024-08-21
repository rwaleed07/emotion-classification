import streamlit as st
import pandas as pd
import numpy as np
import re
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

# Initialize tools
stop_words = {'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 
              'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 
              'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 
              'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 
              'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 
              'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 
              'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 
              'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 
              'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now'}

# Load the model with error handling
model_path = 'emotion_classification_model.h5'  # Replace with the actual path
try:
    model = load_model(model_path)
except Exception as e:
    st.write("Error loading model: ", e)

# Initialize or load tokenizer (update this to match your training setup)
try:
    tokenizer = Tokenizer()  # Define the tokenizer
    # If you have a saved tokenizer, load it here:
    # with open('tokenizer.pickle', 'rb') as handle:
    #     tokenizer = pickle.load(handle)
except Exception as e:
    st.write("Error initializing/loading tokenizer: ", e)

# Function to preprocess text without NLTK
def preprocess_text(text):
    try:
        # Lowercase and remove non-alphabetic characters
        text = text.lower()
        tokens = re.findall(r'\b\w+\b', text)
        # Remove stopwords
        tokens = [word for word in tokens if word not in stop_words]
        return " ".join(tokens)
    except Exception as e:
        st.write("Error in preprocessing: ", e)
        return ""

# Streamlit app
st.title("Emotion Classification App")

st.write("This app classifies emotions from text input.")

# Text input from the user
user_input = st.text_area("Enter a sentence to classify:")

if st.button("Classify"):
    if user_input:
        # Preprocess the input
        processed_text = preprocess_text(user_input)

        try:
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
        except Exception as e:
            st.write("Error during prediction: ", e)
    else:
        st.write("Please enter some text for classification.")



