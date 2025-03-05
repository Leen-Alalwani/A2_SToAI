import streamlit as st
import os
import requests
from bs4 import BeautifulSoup
import numpy as np
import faiss
from mistralai import Mistral

# Set up Mistral API key
api_key = os.getenv("MISTRAL_API_KEY", "kOCiq0K2qXcwVxhh8vKRaC7POzJ5Un2m")

# Function to fetch and parse policy text
def fetch_policy_data(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    tag = soup.find("div")
    return tag.text.strip() if tag else "Policy content not found."

# Function to chunk text
def chunk_text(text, chunk_size=512):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

# Get embeddings
def get_text_embedding(list_txt_chunks):
    client = Mistral(api_key=api_key)
    embeddings_response = client.embeddings.create(model="mistral-embed", inputs=list_txt_chunks)
    return [e.embedding for e in embeddings_response.data]

# Create FAISS index
def create_faiss_index(embeddings):
    if not embeddings:
        return None  # Prevent crashes if no embeddings
    d = len(embeddings[0])
    index = faiss.IndexFlatL2(d)
    index.add(np.array(embeddings))
    return index

# Retrieve relevant chunks
def search_relevant_chunks(faiss_index, query_embedding, k=2):
    if not faiss_index:
        return []  # Handle missing index
    D, I = faiss_index.search(query_embedding, k)
    return I[0]

# Generate AI response
def mistral_answer(query, context):
    prompt = f"""
    Context:
    {context}
    
    Given the context, answer the query:
    {query}
    """
    client = Mistral(api_key=api_key)
    response = client.chat.complete(model="mistral-large-latest", messages=[{"role": "user", "content": prompt}])
    return response.choices[0].message.content

# Streamlit App
def streamlit_app():
    st.title("UDST Policy Chatbot")

    policies = [
        "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/academic-annual-leave-policy",
        "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/academic-appraisal-policy",
        "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/academic-credentials-policy"
    ]

    selected_policy_url = st.selectbox("Select a Policy", policies)
    policy_text = fetch_policy_data(selected_policy_url)
    chunks = chunk_text(policy_text)

    if not chunks:
        st.error("No policy data available.")
        return

    embeddings = get_text_embedding(chunks)
    faiss_index = create_faiss_index(embeddings)
    query = st.text_input("Enter your question:")

    if query:
        query_embedding = np.array([get_text_embedding([query])[0]])
        indices = search_relevant_chunks(faiss_index, query_embedding)
        context = " ".join([chunks[i] for i in indices]) if indices else "No relevant context found."
        answer = mistral_answer(query, context) if context else "I couldn't find relevant information."
        st.text_area("Answer:", answer, height=200)


if __name__ == "__main__":
    streamlit_app()
