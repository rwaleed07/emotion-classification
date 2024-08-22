import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

# Define the header text with custom color using HTML
header_html1 = """
    <h1 style="color: blue;">Welcome to Sentiment Analysis App</h1>
"""

# Render the HTML in Streamlit
st.markdown(header_html1, unsafe_allow_html=True)

# Load the model
model_path = 'emotion_classification_model.h5'  # Replace with your actual path
model = load_model(model_path)

# Compile the model (if you need to use metrics)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
def preprocess_text(text):
    # Remove HTML tags
    text = BeautifulSoup(text, "html.parser").get_text()

    # Convert to lowercase
    text = text.lower()

    # Remove emojis
    text = emoji.demojize(text)

    # Remove special characters and number
    text = re.sub(r'[@#%&*^$£!()-_+={}\[\]:;<>,.?\/\\\'"`~]', '', text)
    
    # Remove special numbers
    text = re.sub(r'\d+', '', text)

    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()

    # Tokenize the text
    words = nltk.word_tokenize(text)

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    stop_words = [word for word in words if word not in stop_words]
    stop_words = ' '.join(stop_words)

    # Lemmatize the words
    lemmatizer = WordNetLemmatizer()
    lemmetize_words = [lemmatizer.lemmatize(word) for word in words]
    
    # Rejoin words to form the final processed text
    processed_text = ' '.join(lemmetize_words)
    
    return processed_text
  
# Making prediction on a processed_text
def predict_sentiment(text):
    # Preprocess the input text
    cleaned_text = preprocess_text(text)
    
    # Predict the sentiment
    prediction = model.predict([cleaned_text])
    return prediction[0]

# Streamlit app
st.write("Enter a text to analyze its sentiment.")

user_input = st.text_area("Enter your text here:")
if st.button("Analyze"):
    if user_input:
        result = predict_sentiment(user_input)
        st.write(f"Sentiment: {result}")
    else:
        st.write("Please enter some text.")
