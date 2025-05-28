## ML Q&A Evaluator App – Pipeline Summary
## App Setup (Streamlit)

Loads UI and features sidebar.
Randomly selects an ML question.

## Web Scraping

Fetches and cleans ML content from websites (Builtin, IBM).
Removes unwanted tags using BeautifulSoup.

## Text Chunking

Splits content into manageable chunks using RecursiveCharacterTextSplitter.

## Embedding + Retrieval

Converts chunks to embeddings using HuggingFace (MiniLM).
Stores in Chroma vector DB.

Retrieves top-3 relevant docs per question.

## Prompt Creation

Formats context + question into a clear prompt.

## LLM Answer Generation

Loads Falcon-7B via pipeline() for text generation.
Uses LangChain to manage input/output.

## User Interaction

Shows Falcon’s answer.
Lets user input their own answer.

## Similarity Scoring

Embeds both answers and compares using cosine similarity.

## Feedback based on score:
 80–100: Excellent
 50–80: Needs improvement
 <50: Try again

#  Tech Stack
LLM: Falcon-7B
Embeddings: MiniLM-L6-v2
Vector DB: Chroma
Frameworks: Streamlit, LangChain, Hugging Face

