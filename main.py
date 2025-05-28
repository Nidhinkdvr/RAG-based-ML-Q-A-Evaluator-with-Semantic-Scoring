import streamlit as st
import random
import requests
from bs4 import BeautifulSoup

from transformers import pipeline
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# Load and clean text from webpages
@st.cache_data(show_spinner=False)
def fetch_web_content(urls):
    all_text = ""
    for url in urls:
        try:
            response = requests.get(url)
            soup = BeautifulSoup(response.text, "html.parser")
            for tag in soup(["script", "style"]):
                tag.decompose()
            all_text += soup.get_text(separator=" ", strip=True) + "\n"
        except Exception as e:
            st.warning(f"Failed to fetch {url}: {e}")
    return all_text


# Initialize Falcon model using HuggingFace pipeline

@st.cache_resource(show_spinner=True)
def load_falcon_model():
    return pipeline(
        "text-generation",
        model="tiiuae/falcon-7b",
        device=0,
        max_new_tokens=300
    )

# ---------------------------
# Streamlit UI Setup
# ---------------------------
st.set_page_config("ML Q&A Evaluator", layout="wide")
st.title(" Machine Learning Q&A Evaluator")

with st.sidebar:
    st.header("⚙️ Features")
    st.markdown("""
    - Random ML questions  
    - Falcon-7B AI answer  
    - User input evaluation  
    - Semantic similarity scoring
    """)

questions = [
    "What is supervised learning?",
    "Define overfitting in machine learning.",
    "What are neural networks?",
    "Compare regression and classification.",
    "Role of a data scientist?"
]

# Choose a random question
if "question" not in st.session_state:
    st.session_state.question = random.choice(questions)
    st.session_state.qid = random.randint(1000, 9999)

st.subheader(f"Question ID: {st.session_state.qid}")
st.markdown(f"### ? {st.session_state.question}")

# Load Falcon model
falcon = load_falcon_model()

# ---------------------------
# Generate AI Answer
# ---------------------------
if st.button("Generate Falcon Answer"):
    with st.spinner("Falcon is thinking..."):
        context_urls = [
            "https://builtin.com/machine-learning",
            "https://www.ibm.com/topics/machine-learning"
        ]
        context = fetch_web_content(context_urls)
        prompt = f"Context:\n{context}\n\nQuestion:\n{st.session_state.question}\n\nAnswer:"
        result = falcon(prompt)[0]["generated_text"]
        answer_start = result.find("Answer:") + len("Answer:")
        st.session_state.ai_answer = result[answer_start:].strip()
        st.success("Answer generated!")

# ---------------------------
# Display and Compare Answers
# ---------------------------
if "ai_answer" in st.session_state:
    st.markdown("####  Falcon's Answer")
    st.info(st.session_state.ai_answer)

    st.markdown("####  Your Answer")
    user_input = st.text_area("Write your answer here:", height=150)

    if st.button("Evaluate My Answer"):
        if user_input.strip():
            with st.spinner("Comparing answers..."):
                embedder = SentenceTransformer("all-MiniLM-L6-v2")
                ai_embedding = embedder.encode([st.session_state.ai_answer])
                user_embedding = embedder.encode([user_input])
                score = cosine_similarity(ai_embedding, user_embedding)[0][0] * 100

            st.markdown(f"###  Similarity Score: **{score:.2f} / 100**")
            if score > 80:
                st.success("Excellent! Your answer is very close.")
            elif score > 50:
                st.info("Decent effort, but there's room to improve.")
            else:
                st.warning("Your answer is quite different. Try revising.")
        else:
            st.warning("Please write an answer first.")
