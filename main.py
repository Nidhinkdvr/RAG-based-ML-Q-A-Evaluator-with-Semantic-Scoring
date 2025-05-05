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
import torch
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Fetch and clean articles
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
            print(f"Error fetching {url}: {e}")
    return documents

urls = [
    "https://builtin.com/machine-learning",
    "https://www.analyticsvidhya.com/blog/category/data-science/",
    "https://www.ibm.com/topics/machine-learning",
]

data = fetch_articles(urls)

#  Chunking
splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
docs = splitter.split_documents(data)

# Embeddings & Vector DB
embedding = HuggingFaceEmbeddings()
db = Chroma.from_documents(docs, embedding)
retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 3})

#  RAG Prompt Setup
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

#  Generate Question and ID
question_bank = [
    "What is supervised learning?",
    "Explain overfitting in ML.",
    "What is the role of a data scientist?",
    "Difference between classification and regression?",
    "How does a neural network work?"
]

qid = f"Q{random.randint(1000, 9999)}"
question = random.choice(question_bank)
print(f"{qid}: {question}")

#  RAG answer
generated_answer = rag_chain.invoke(question).replace("</s>", "").strip()
print("\nGenerated Answer:\n", generated_answer)

#  User input
user_answer = input("\nYour Answer:\n")

# Similarity Scoring
similarity_model = SentenceTransformer('all-MiniLM-L6-v2')
gen_vec = similarity_model.encode([generated_answer])
user_vec = similarity_model.encode([user_answer])
score = cosine_similarity(gen_vec, user_vec)[0][0] * 100

print(f"\nSimilarity Score: {score:.2f}/100")
