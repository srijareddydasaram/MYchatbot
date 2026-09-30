
import streamlit as st
import requests

st.title("🤖 My  AI Chatbot")

user_input = st.chat_input("Type your message...")

if user_input:
    st.write("You:", user_input)

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "llama3.2",
            "prompt": user_input,
            "stream": False
        }
    )

    answer = response.json()["response"]

    st.write("Bot:", answer)