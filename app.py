import streamlit as st
import numpy as np
import faiss
import os
import pickle
from mistralai import Mistral, UserMessage
import dotenv

# Load environment variables
dotenv.load_dotenv()

# Page configuration
st.set_page_config(page_title="RAG Knowledge Assistant", page_icon="🧠", layout="wide")

# Initialize session state
if 'api_key' not in st.session_state:
    st.session_state.api_key = os.getenv("MISTRAL_API_KEY", "")
if 'chunks' not in st.session_state:
    st.session_state.chunks = []
if 'sources' not in st.session_state:
    st.session_state.sources = []
if 'index' not in st.session_state:
    st.session_state.index = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

def get_text_embedding(list_txt_chunks):
    client = Mistral(api_key=st.session_state.api_key)
    embeddings_batch_response = client.embeddings.create(model="mistral-embed", inputs=list_txt_chunks)
    return embeddings_batch_response.data

def load_existing_index():
    try:
        index = faiss.read_index("assets/rag_index.faiss")
        with open("assets/rag_data.pkl", "rb") as f:
            data = pickle.load(f)
        st.session_state.chunks = data["chunks"]
        st.session_state.sources = data["sources"]
        st.session_state.index = index
        st.success("Successfully loaded existing knowledge base.")
        return True
    except Exception as e:
        st.error(f"Failed to load existing index: {e}")
        return False

def rag_query(question, k=3):
    if not st.session_state.index or not st.session_state.chunks:
        return "Knowledge Base Not Loaded", ""

    question_embeddings = get_text_embedding([question])
    if not question_embeddings or question_embeddings[0] is None:
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
            # Add more sources as needed
        ]
        for description, url in sources:
            st.markdown(f"[{description}]({url})")

        st.divider()
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()

def main():
    display_sidebar()
    load_existing_index()

    st.markdown("# 🧠 RAG Knowledge Assistant")
    st.markdown("Welcome to the RAG Knowledge Assistant! Ask questions below.")

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
                response = "Please add knowledge sources before asking questions."
            else:
                response, context = rag_query(user_question)
                st.markdown(response)
                if context:
                    with st.expander("View retrieved context"):
                        st.markdown(context)

            st.session_state.chat_history.append({"role": "assistant", "content": response, "context": context})

if __name__ == "__main__":
    main()
