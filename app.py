import requests
from bs4 import BeautifulSoup
import numpy as np
import faiss
import os
from mistralai import Mistral

# ✅ Set up API Key
MISTRAL_API_KEY = "kOCiq0K2qXcwVxhh8vKRaC7POzJ5Un2m"  # Replace with your actual key
os.environ["MISTRAL_API_KEY"] = MISTRAL_API_KEY

# ✅ Initialize Mistral Client
client = Mistral(api_key=MISTRAL_API_KEY)

# ✅ List of UDST Policy URLs
policy_urls = [
    "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/academic-annual-leave-policy",
    "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/academic-appraisal-policy",
    "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/academic-appraisal-procedure",
    "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/academic-credentials-policy",
    "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/academic-freedom-policy",
    "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/academic-members%E2%80%99-retention-policy",
    "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/academic-qualifications-policy",
    "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/credit-hour-policy",
    "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/intellectual-property-policy",
    "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/joint-appointment-policy"
]

# ✅ Function to fetch policy text from a given URL
def fetch_policy_data(url):
    response = requests.get(url)
    if response.status_code != 200:
        print(f"⚠️ Failed to retrieve policy data from {url}")
        return ""

    soup = BeautifulSoup(response.text, "html.parser")
    tag = soup.find("div")
    return tag.text.strip() if tag else "⚠️ Policy content not found."

# ✅ Chunk text into smaller parts (for embedding)
def chunk_text(text, chunk_size=512):
    return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]

# ✅ Get embeddings using Mistral
def get_text_embedding(text_chunks):
    embeddings_batch_response = client.embeddings.create(model="mistral-embed", inputs=text_chunks)
    return embeddings_batch_response.data

# ✅ Create FAISS index for searching policy chunks
def create_faiss_index(embeddings):
    d = len(embeddings[0].embedding)  # Embedding dimension
    index = faiss.IndexFlatL2(d)
    embedding_vectors = np.array([emb.embedding for emb in embeddings])
    index.add(embedding_vectors)
    return index

# ✅ Search FAISS for the most relevant policy chunks
def search_relevant_chunks(index, query_embedding, k=2):
    D, I = index.search(query_embedding, k)
    return I.flatten()

# ✅ Main Execution
if __name__ == "__main__":
    all_chunks = []  # Stores all text chunks from multiple policies
    all_chunk_sources = []  # Tracks the source of each chunk

    # ✅ Process all policies
    for url in policy_urls:
        print(f"📌 Fetching policy from: {url}")
        policy_text = fetch_policy_data(url)

        if policy_text:  # Proceed only if valid text is found
            chunks = chunk_text(policy_text)
            all_chunks.extend(chunks)
            all_chunk_sources.extend([url] * len(chunks))  # Track the source of each chunk

    # ✅ Generate embeddings and create FAISS index
    text_embeddings = get_text_embedding(all_chunks)
    faiss_index = create_faiss_index(text_embeddings)

    # ✅ Example Query
    question = "What is the attendance policy for students at UDST?"
    question_embedding = np.array([get_text_embedding([question])[0].embedding])

    # Retrieve relevant chunks
    relevant_chunk_ids = search_relevant_chunks(faiss_index, question_embedding)
    retrieved_chunks = [all_chunks[i] for i in relevant_chunk_ids]
    retrieved_sources = [all_chunk_sources[i] for i in relevant_chunk_ids]

    # ✅ Display retrieved policy information
    print("\n📌 Retrieved Policy Info:")
    for i in range(len(retrieved_chunks)):
        print(f"\n🔹 **Source:** {retrieved_sources[i]}")
        print(f"🔹 **Extracted Policy Info:** {retrieved_chunks[i][:500]}...")  # Display first 500 characters
