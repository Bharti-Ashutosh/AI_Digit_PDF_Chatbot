import streamlit as st
import numpy as np
from PIL import Image
import cv2

from digit_model import predict_digit
from pdf_reader import extract_text
from chatbot import chatbot_answer

st.title("🧠 AI Digit Recognition + PDF Chatbot")

st.header("✍️ Handwritten Digit Recognition")
img_file = st.file_uploader("Upload digit image", type=["png","jpg","jpeg"])

if img_file:
    image = Image.open(img_file)
    st.image(image, width=150)
    img = np.array(image)
    digit = predict_digit(img)
    st.success(f"Predicted Digit: {digit}")

st.header("📄 PDF Reader & Chatbot")
pdf_file = st.file_uploader("Upload PDF", type=["pdf"])

if pdf_file:
    text = extract_text(pdf_file)
    st.text_area("Extracted Text", text[:1000])

    question = st.text_input("Ask a question from PDF")
    if question:
        answer = chatbot_answer(question, text)
        st.info(f"Answer: {answer}")
