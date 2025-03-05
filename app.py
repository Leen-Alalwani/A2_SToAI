import streamlit as st
import numpy as np
import faiss
import os
import pickle
import time
from mistralai import Mistral, UserMessage
import dotenv
from bs4 import BeautifulSoup
import requests

# Load environment variables
dotenv.load_dotenv()

# Page configuration
st.set_page_config(page_title="UDST Policy Assistant", page_icon="🧠", layout="wide")

# Initialize session state
if 'api_key' not in st.session_state:
    st.session_state.api_key = os.getenv("MISTRAL_API_KEY", "kOCiq0K2qXcwVxhh8vKRaC7POzJ5Un2m")
if 'chunks' not in st.session_state:
    st.session_state.chunks = []
if 'sources' not in st.session_state:
    st.session_state.sources = []
if 'index' not in st.session_state:
    st.session_state.index = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

def get_text_embedding(list_txt_chunks, batch_size=10):
    client = Mistral(api_key=st.session_state.api_key)
    all_embeddings = []
    for i in range(0, len(list_txt_chunks), batch_size):
        batch = list_txt_chunks[i:i+batch_size]
        try:
            embeddings_batch_response = client.embeddings.create(model="mistral-embed", inputs=batch)
            all_embeddings.extend(embeddings_batch_response.data)
            time.sleep(0.5)
        except Exception as e:
            st.error(f"Error getting embeddings: {e}")
            return []
    return all_embeddings

def load_policy(url):
    response = requests.get(url)
    html_doc = response.text
    soup = BeautifulSoup(html_doc, "html.parser")
    text = soup.get_text()
    chunk_size = 512
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
    return chunks

def create_index(chunks):
    embeddings = get_text_embedding(chunks)
    if not embeddings:
        return None
    d = len(embeddings[0].embedding)
    index = faiss.IndexFlatL2(d)
    index.add(np.array([emb.embedding for emb in embeddings]))
    return index

def rag_query(question, k=3):
    if not st.session_state.index or not st.session_state.chunks:
        return "Knowledge Base Not Loaded", ""

    question_embeddings = get_text_embedding([question])
    if not question_embeddings:
        return "Error: Could not generate embeddings for the question.", ""

    query_embedding = np.array([question_embeddings[0].embedding])
    D, I = st.session_state.index.search(query_embedding, k=k)
    retrieved_chunks = [st.session_state.chunks[i] for i in I.tolist()[0]]

    context = "\n---\n".join(retrieved_chunks)
    prompt = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"

    client = Mistral(api_key=st.session_state.api_key)
    messages = [UserMessage(content=prompt)]
    try:
        chat_response = client.chat.complete(model="mistral-large-latest", messages=messages)
        response = chat_response.choices[0].message.content
    except Exception as e:
        response = f"Error generating response: {str(e)}"

    return response, context

def display_sidebar():
    with st.sidebar:
        st.image("https://www.udst.edu.qa/themes/custom/cnaq/logo.png", width=200)
        st.title("📚 Config")

        api_key = st.text_input("Mistral API Key", value=st.session_state.api_key, type="password")
        if api_key != st.session_state.api_key:
            st.session_state.api_key = api_key

        st.divider()
        st.subheader("Knowledge Sources")
        sources = [
            ("Sport and Wellness", "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/sport-and-wellness-facilities-and"),
            ("Attendance Policy", "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/student-attendance-policy"),
            
        ]
        selected_policy = st.selectbox("Select a policy:", options=list(zip(*sources))[1])
        if st.button("Load Policy"):
            st.session_state.chunks = load_policy(selected_policy)
            st.session_state.index = create_index(st.session_state.chunks)
            st.success("Policy loaded successfully.")

        st.divider()
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()

def main():
    display_sidebar()

    st.markdown("# 🧠 UDST Policy Assistant")
    st.markdown("Welcome to the UDST Policy Assistant! Ask questions below about university policies.")

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("context") and message["role"] == "assistant":
                with st.expander("View retrieved context"):
                    st.markdown(message["context"])

    user_question = st.chat_input("Ask a question about university policies...")
    if user_question:
        st.session_state.chat_history.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)

        with st.chat_message("assistant"):
            if not st.session_state.api_key:
                response = "Please enter your Mistral API key in the sidebar."
            elif not st.session_state.index:
                response = "Please load a policy before asking questions."
            else:
                response, context = rag_query(user_question)
                st.markdown(response)
                if context:
                    with st.expander("View retrieved context"):
                        st.markdown(context)

            st.session_state.chat_history.append({"role": "assistant", "content": response, "context": context})

if __name__ == "__main__":
    main()
