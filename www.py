{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "b7cc43f5-5881-406c-a9f2-4cd9cf6d5d9b",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2024-08-21 00:55:51.785673: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.\n",
      "To enable the following instructions: AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.\n"
     ]
    }
   ],
   "source": [
    "import streamlit as st\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import nltk\n",
    "from nltk.corpus import stopwords\n",
    "from nltk.tokenize import word_tokenize\n",
    "from nltk.stem import WordNetLemmatizer\n",
    "from tensorflow.keras.models import load_model\n",
    "from tensorflow.keras.preprocessing.sequence import pad_sequences\n",
    "import pickle"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "3b344950-84a0-416f-94af-aab844b4664c",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "[nltk_data] Downloading package punkt to /Users/app/nltk_data...\n",
      "[nltk_data]   Package punkt is already up-to-date!\n",
      "[nltk_data] Downloading package stopwords to /Users/app/nltk_data...\n",
      "[nltk_data]   Package stopwords is already up-to-date!\n",
      "[nltk_data] Downloading package wordnet to /Users/app/nltk_data...\n",
      "[nltk_data]   Package wordnet is already up-to-date!\n",
      "WARNING:absl:Compiled the loaded model, but the compiled metrics have yet to be built. `model.compile_metrics` will be empty until you train or evaluate the model.\n",
      "2024-08-21 00:58:53.623 \n",
      "  \u001b[33m\u001b[1mWarning:\u001b[0m to view this Streamlit app on a browser, run it with the following\n",
      "  command:\n",
      "\n",
      "    streamlit run /Applications/anaconda3/lib/python3.12/site-packages/ipykernel_launcher.py [ARGUMENTS]\n"
     ]
    }
   ],
   "source": [
    "\n",
    "# Download necessary NLTK data\n",
    "nltk.download('punkt')\n",
    "nltk.download('stopwords')\n",
    "nltk.download('wordnet')\n",
    "\n",
    "# Initialize tools\n",
    "lemmatizer = WordNetLemmatizer()\n",
    "stop_words = set(stopwords.words('english'))\n",
    "\n",
    "# Load the model\n",
    "model_path = 'emotion_classification_model.h5'  # Replace with the actual path\n",
    "model = load_model(model_path)\n",
    "\n",
    "# Load the tokenizer\n",
    "with open('tokenizer.pickle', 'rb') as handle:\n",
    "    tokenizer = pickle.load(handle)\n",
    "\n",
    "# Function to preprocess text\n",
    "def preprocess_text(text):\n",
    "    text = text.lower()\n",
    "    tokens = word_tokenize(text)\n",
    "    tokens = [lemmatizer.lemmatize(word) for word in tokens if word.isalpha() and word not in stop_words]\n",
    "    return \" \".join(tokens)\n",
    "\n",
    "# Streamlit app\n",
    "st.title(\"Emotion Classification App\")\n",
    "\n",
    "st.write(\"This app classifies emotions from text input.\")\n",
    "\n",
    "# Text input from the user\n",
    "user_input = st.text_area(\"Enter a sentence to classify:\")\n",
    "\n",
    "if st.button(\"Classify\"):\n",
    "    if user_input:\n",
    "        # Preprocess the input\n",
    "        processed_text = preprocess_text(user_input)\n",
    "\n",
    "        # Convert text to sequence\n",
    "        text_seq = tokenizer.texts_to_sequences([processed_text])\n",
    "        max_len = 100  # Adjust based on training\n",
    "        text_pad = pad_sequences(text_seq, maxlen=max_len)\n",
    "\n",
    "        # Predict the emotion\n",
    "        prediction = model.predict(text_pad)\n",
    "        predicted_class = np.argmax(prediction, axis=1)[0]\n",
    "\n",
    "        # Define emotion labels (adjust based on your dataset)\n",
    "        emotion_labels = ['anger', 'love', 'fear', 'joy', 'sadness', 'surprise']  # Update according to your dataset\n",
    "        predicted_emotion = emotion_labels[predicted_class]\n",
    "\n",
    "        # Display the result\n",
    "        st.write(f\"The predicted emotion is: **{predicted_emotion.capitalize()}**\")\n",
    "    else:\n",
    "        st.write(\"Please enter some text for classification.\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "de8051c4-a3bb-49f9-8555-37899df792ba",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
