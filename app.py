import streamlit as st
import os
import requests
from bs4 import BeautifulSoup
import numpy as np
from mistralai import Mistral
import faiss
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Securely load API key (recommend using python-dotenv)
api_key = os.getenv("kOCiq0K2qXcwVxhh8vKRaC7POzJ5Un2m")
if not api_key:
    st.error("Mistral API Key not found. Please set MISTRAL_API_KEY environment variable.")
    st.stop()

# Fetching and parsing policy data with error handling
def fetch_policy_data(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        
        # More robust text extraction
        text_elements = soup.find_all(['p', 'div', 'article'])
        text = " ".join([elem.get_text(strip=True) for elem in text_elements])
        
        if not text:
            st.warning(f"No text extracted from {url}")
            return "No policy text found."
        
        return text
    except requests.RequestException as e:
        logger.error(f"Error fetching policy from {url}: {e}")
        st.error(f"Could not fetch policy from {url}")
        return f"Error: {str(e)}"

# Rest of the functions remain the same as in the original script

def streamlit_app():
    st.title('UDST Policies Q&A')
    st.markdown("*Retrieve policy information using natural language queries*")

    policies = [
        "Registration Policy",
        "Examination Policy", 
        "Graduation Policy",
        "Admissions Policy",
        "Transfer Policy",
        
    ]

    policy_urls = [
        "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/registration-policy",
        "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/examination-policy",
        
    ]

    selected_policy = st.selectbox("Select a Policy", policies)
    selected_index = policies.index(selected_policy)
    selected_policy_url = policy_urls[selected_index]

    try:
        policy_text = fetch_policy_data(selected_policy_url)
        chunks = chunk_text(policy_text)

        # Generate embeddings for the chunks and create a FAISS index
        embeddings = get_text_embedding(chunks)
        faiss_index = create_faiss_index(embeddings)

        # Input box for query
        query = st.text_input(f"Ask a question about the {selected_policy}:")

        if query:
            # Embed the user query and search for relevant chunks
            query_embedding = np.array([get_text_embedding([query])[0].embedding])
            I = search_relevant_chunks(faiss_index, query_embedding, k=2)
            retrieved_chunks = [chunks[i] for i in I.tolist()[0]]
            context = " ".join(retrieved_chunks)

            # Generate answer from the model
            answer = mistral_answer(query, context)

            # Display the answer
            st.text_area("Answer:", answer, height=200)

    except Exception as e:
        st.error(f"An error occurred: {e}")
        logger.error(f"Unexpected error: {e}")

if __name__ == '__main__':
    streamlit_app()
