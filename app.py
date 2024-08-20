{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "55ece107-1f08-464e-9d1d-977e7374850d",
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
      "2024-08-20 21:26:51.345 \n",
      "  \u001b[33m\u001b[1mWarning:\u001b[0m to view this Streamlit app on a browser, run it with the following\n",
      "  command:\n",
      "\n",
      "    streamlit run /Applications/anaconda3/lib/python3.12/site-packages/ipykernel_launcher.py [ARGUMENTS]\n"
     ]
    }
   ],
   "source": [
    "import streamlit as st\n",
    "import nltk\n",
    "import numpy as np\n",
    "from nltk.corpus import stopwords\n",
    "from nltk.tokenize import word_tokenize\n",
    "from nltk.stem import WordNetLemmatizer\n",
    "from tensorflow.keras.models import load_model\n",
    "from tensorflow.keras.preprocessing.sequence import pad_sequences\n",
    "import pickle\n",
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
    "# Load the pre-trained model\n",
    "model = load_model('emotion_classification_model.h5')\n",
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
    "        \n",
    "        # Convert text to sequence\n",
    "        text_seq = tokenizer.texts_to_sequences([processed_text])\n",
    "        max_len = 100  # Adjust based on training\n",
    "        text_pad = pad_sequences(text_seq, maxlen=max_len)\n",
    "        \n",
    "        # Predict the emotion\n",
    "        prediction = model.predict(text_pad)\n",
    "        predicted_class = np.argmax(prediction, axis=1)[0]\n",
    "        \n",
    "        # Define emotion labels (adjust based on your dataset)\n",
    "        emotion_labels = ['anger', 'love' , 'joy', 'sadness', 'surprise', 'fear']  # Update according to your dataset\n",
    "        predicted_emotion = emotion_labels[predicted_class]\n",
    "        \n",
    "        # Display the result\n",
    "        st.write(f\"The predicted emotion is: **{predicted_emotion.capitalize()}**\")\n",
    "    else:\n",
    "        st.write(\"Please enter some text for classification.\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "f9c22ea2-0728-4579-8f5a-7a93965d749a",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "1f96aaad-16a4-41e1-b3bf-285c00bfcb11",
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
