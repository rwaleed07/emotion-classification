import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load the tokenizer and model
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    return tokenizer, model

tokenizer, model = load_model()

# Title of the app
st.title("Story Generator")

# Input prompt from the user
prompt = st.text_input("Enter a prompt to start your story:")

# Generate the story
if prompt:
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(inputs.input_ids, max_length=200, num_return_sequences=1)
    story = tokenizer.decode(outputs[0], skip_special_tokens=True)
    st.write("### Your Generated Story:")
    st.write(story)
