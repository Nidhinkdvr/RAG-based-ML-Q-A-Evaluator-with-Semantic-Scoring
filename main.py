import streamlit as st
import random
import requests
from bs4 import BeautifulSoup
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFacePipeline
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# CACHE FUNCTIONS 
@st.cache_data(show_spinner=False)
def fetch_articles(urls):
    documents = []
    for url in urls:
        try:
            res = requests.get(url)
            soup = BeautifulSoup(res.text, 'html.parser')
            for tag in soup(['script', 'style']):
                tag.decompose()
            text = ' '.join(soup.get_text().split())
            documents.append(Document(page_content=text, metadata={"source": url}))
        except Exception as e:
            st.warning(f"Error fetching {url}: {e}")
    return documents

@st.cache_resource(show_spinner=True)
def setup_pipeline():
    urls = [
        "https://builtin.com/machine-learning",
        "https://www.analyticsvidhya.com/blog/category/data-science/",
        "https://www.ibm.com/topics/machine-learning",
    ]
    data = fetch_articles(urls)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
    docs = splitter.split_documents(data)
    embedding = HuggingFaceEmbeddings()
    db = Chroma.from_documents(docs, embedding)
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    model_id = "tiiuae/falcon-7b"
    pipe = pipeline("text-generation", model=model_id, device=0, max_new_tokens=300)
    llm = HuggingFacePipeline(pipeline=pipe)

    prompt_template = """
    Answer the question based on this context:
    {context}

    Question:
    {question}

    Answer:
    """
    prompt = PromptTemplate(input_variables=["context", "question"], template=prompt_template)
    llm_chain = prompt | llm | StrOutputParser()
    rag_chain = {"context": retriever, "question": RunnablePassthrough()} | llm_chain

    return rag_chain

# STREAMLIT UI 
st.set_page_config(page_title="ML Q&A Evaluator", page_icon="🧠", layout="wide")
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>🧠 Machine Learning Q&A Evaluator</h1>", unsafe_allow_html=True)

with st.sidebar:
    st.header("🎯 App Features")
    st.markdown("""
    - ✅ Random ML Question Generator  
    - 💬 RAG-based Answer Generator  
    - 🧪 User Answer Evaluation  
    - 🔍 Semantic Similarity Scoring  
    """)
    st.info("Built using LangChain, HuggingFace, Falcon-7B")

# QUESTION GENERATION 
question_bank = [
    "What is supervised learning?",
    "Explain overfitting in ML.",
    "What is the role of a data scientist?",
    "Difference between classification and regression?",
    "How does a neural network work?"
]

if "qid" not in st.session_state:
    st.session_state.qid = f"Q{random.randint(1000, 9999)}"
    st.session_state.question = random.choice(question_bank)

st.subheader(f" Question ID: `{st.session_state.qid}`")
st.markdown(f"<h3 style='color:#FF6F61;'>❓ {st.session_state.question}</h3>", unsafe_allow_html=True)

# RAG PIPELINE & ANSWER 
rag_chain = setup_pipeline()

if st.button("Generate Answer"):
    with st.spinner("Generating context-based answer..."):
        generated_answer = rag_chain.invoke(st.session_state.question).replace("</s>", "").strip()
        st.session_state.generated_answer = generated_answer
        st.success("Answer Generated Successfully! ")

if "generated_answer" in st.session_state:
    st.markdown("###  Generated Answer")
    st.markdown(f"<div style='background-color:#f0f8ff; padding:15px; border-radius:10px;'>{st.session_state.generated_answer}</div>", unsafe_allow_html=True)

    st.markdown("### 📝 Your Answer")
    user_input = st.text_area("Write your answer below ", height=150, placeholder="Type your answer here...")

    if st.button(" Evaluate Answer"):
        if not user_input.strip():
            st.warning("Please type an answer to evaluate.")
        else:
            with st.spinner("Calculating similarity score..."):
                model = SentenceTransformer("all-MiniLM-L6-v2")
                gen_vec = model.encode([st.session_state.generated_answer])
                user_vec = model.encode([user_input])
                score = cosine_similarity(gen_vec, user_vec)[0][0] * 100

            st.markdown(f"<h3 style='color:#2196F3;'>Similarity Score: {score:.2f} / 100</h3>", unsafe_allow_html=True)

            if score > 80:
                st.success("Excellent! Your answer closely matches the generated response. 🎉")
            elif score > 50:
                st.info("Good effort! There is some similarity, but room for improvement.")
            else:
                st.warning("Low similarity. Try improving your answer with more specific details.")
