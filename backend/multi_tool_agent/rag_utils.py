import os
from typing import List
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss


def extract_pdf_text(pdf_path: str) -> str:
    """Extracts all text from a PDF file."""
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text() or ""
        text += page_text
    return text


def chunk_text(text: str, chunk_size: int = 500) -> List[str]:
    """Splits text into chunks for embedding."""
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]


def create_embeddings(chunks: List[str], model_name: str = 'all-MiniLM-L6-v2') -> np.ndarray:
    """Creates embeddings for a list of text chunks."""
    model = SentenceTransformer(model_name)
    embeddings = model.encode(chunks, show_progress_bar=True)
    return np.array(embeddings).astype('float32')

# --------- FAISS Index Operations ----------
def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatL2:
    """Initializes and populates a FAISS index from embeddings."""
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

def query_faiss(query: str, chunks: List[str], index: faiss.IndexFlatL2, model, top_k: int = 3) -> List[str]:
    """Retrieves top-K relevant text chunks for a query."""
    query_emb = model.encode([query])
    D, I = index.search(np.array(query_emb).astype('float32'), top_k)
    # I is shape (num_queries, top_k), usually (1, top_k): flatten before indexing
    indices = I[0]  # Get the first (and only) row for single query
    return [chunks[i] for i in indices]


# --------- Pipeline Helper ---------
def build_pipeline(pdf_path: str, chunk_size: int = 500, model_name: str = 'all-MiniLM-L6-v2'):
    """Returns chunks, embeddings array, FAISS index and embedding model."""
    # 1. Extract text
    text = extract_pdf_text(pdf_path)
    # 2. Chunk text
    chunks = chunk_text(text, chunk_size)
    # 3. Create embeddings
    model = SentenceTransformer(model_name)
    embeddings = model.encode(chunks, show_progress_bar=True)
    embeddings = np.array(embeddings).astype('float32')
    # 4. Build FAISS index
    index = build_faiss_index(embeddings)
    return chunks, index, model

