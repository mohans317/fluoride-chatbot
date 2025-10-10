import streamlit as st
import google.generativeai as genai

genai.configure(api_key=st.secrets["GEMINI_API_KEY"])  # Replace with your actual key

st.title("* Fluoride Awareness Chatbot *")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Capture prompt from user
prompt = st.chat_input("Ask a question about fluoride...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Compose concise instruction for Gemini
    instruction = (
        "Answer the following question in one short sentence or maximum four sentences. "
        "Skip extra details unless specifically asked. "
        "Question: " + prompt
    )

    model = genai.GenerativeModel("gemini-2.5-pro")
    with st.spinner("Generating answer..."):
        response = model.generate_content(instruction)
    answer = response.text

    st.session_state.messages.append({"role": "assistant", "content": answer})

# Display chat history immediately, including the latest user and assistant messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])








