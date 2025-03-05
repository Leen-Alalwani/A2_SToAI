import streamlit as st
import os
import requests
from bs4 import BeautifulSoup
import numpy as np
from mistralai import Mistral
import faiss

# Set up your Mistral API key
os.environ["MISTRAL_API_KEY"] = "kOCiq0K2qXcwVxhh8vKRaC7POzJ5Un2m"
api_key = os.getenv("MISTRAL_API_KEY")

# Fetching and parsing policy data
def fetch_policy_data(url):
    response = requests.get(url)
    html_doc = response.text
    soup = BeautifulSoup(html_doc, "html.parser")
    tag = soup.find("div")
    text = tag.text.strip()
    return text

# Chunking function to break text into smaller parts
def chunk_text(text, chunk_size=512):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

# Get embeddings for text chunks
def get_text_embedding(list_txt_chunks):
    client = Mistral(api_key=api_key)
    embeddings_batch_response = client.embeddings.create(model="mistral-embed", inputs=list_txt_chunks)
    return embeddings_batch_response.data


# Initialize FAISS index
def create_faiss_index(embeddings):
    embedding_vectors = np.array([embedding.embedding for embedding in embeddings])
    d = embedding_vectors.shape[1]
    index = faiss.IndexFlatL2(d)
    faiss_index = faiss.IndexIDMap(index)
    faiss_index.add_with_ids(embedding_vectors, np.array(range(len(embedding_vectors))))
    return faiss_index

# Search for the most relevant chunks based on query embedding
def search_relevant_chunks(faiss_index, query_embedding, k=2):
    D, I = faiss_index.search(query_embedding, k)
    return I


# Mistral model to generate answers based on context
def mistral_answer(query, context):
    prompt = f"""
    Context information is below.
    ---------------------
    {context}
    ---------------------
    Given the context information and not prior knowledge, answer the query.
    Query: {query}
    Answer:
    """
    client = Mistral(api_key=api_key)
    messages = [{"role": "user", "content": prompt}]
    chat_response = client.chat.complete(model="mistral-large-latest", messages=messages)
    return chat_response.choices[0].message.content

# Streamlit Interface
def streamlit_app():
    st.title('UDST Policies Q&A')

    # List of policy URLs
     # List of policies with descriptive names
    policies = [
        ("Registration Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/registration-policy"),
        ("Examination Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/examination-policy"),
        ("Graduation Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/udst-policies-and-procedures/graduation-policy"),
        ("Admissions Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/admissions-policy"),
        ("Transfer Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/transfer-policy"),
        ("International Student Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/udst-policies-and-procedures/international-student-policy"),
        ("Academic Standing Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/academic-standing-policy"),
        ("Intellectual Property Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/intellectual-property-policy"),
        ("Credit Hour Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/credit-hour-policy"),
        ("Student Conduct Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/student-conduct-policy"),
        ("Student Appeals Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/student-appeals-policy"),
        ("Academic Appraisal Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/academic-appraisal-policy"),
        ("Student Engagement Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/udst-policies-and-procedures/student-engagement-policy"),
        ("Academic Credentials Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/academic-credentials-policy"),
        ("Graduate Admissions Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/graduate-admissions-policy"),
        ("Student Attendance Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/student-attendance-policy"),
        ("Final Grade Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/final-grade-policy"),
        ("Academic Freedom Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/academic-freedom-policy"),
        ("Academic Qualifications Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/academic-qualifications-policy"),
        ("Joint Appointment Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/joint-appointment-policy"),
        ("Program Accreditation Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/program-accreditation-policy"),
        ("Academic Schedule Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/udst-policies-and-procedures/academic-schedule-policy"),
        ("Graduate Academic Standing Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/udst-policies-and-procedures/graduate-academic-standing-policy"),
        ("Academic Members’ Retention Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/academic-members%E2%80%99-retention-policy"),
        ("Academic Professional Development": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/academic-professional-development"),
        ("Academic Annual Leave Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/academic-annual-leave-policy"),
        ("Graduate Final Grade Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/udst-policies-and-procedures/graduate-final-grade-policy"),
        ("Student Counselling Services Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/udst-policies-and-procedures/student-counselling-services-policy"),
        ("Scholarship and Financial Assistance": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/scholarship-and-financial-assistance"),
        ("Use of Library Space Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/use-library-space-policy"),
        ("Sport and Wellness Facilities": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/sport-and-wellness-facilities-and")
    ]

    # Select a policy from the list
    selected_policy_url = st.selectbox('Select a Policy', policies)

    # Fetch policy data and chunk it
    policy_text = fetch_policy_data(selected_policy_url)
    chunks = chunk_text(policy_text)

    # Generate embeddings for the chunks and create a FAISS index
    embeddings = get_text_embedding(chunks)
    faiss_index = create_faiss_index(embeddings)

    # Input box for query
    query = st.text_input("Enter your Query:")

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

# Run Streamlit app
if __name__ == '__main__':
    streamlit_app()
