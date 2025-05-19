import tempfile
import os
import uuid
import re
import requests
import pandas as pd
import fitz  # PyMuPDF
import gradio as gr
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from langchain.schema import Document
import matplotlib.pyplot as plt
import numpy as np

# ---------- Configuration ----------
CSV_PATH = "/Users/jigyanshupati/semantic_researc_paper/arXiv_scientific_dataset_cleaned.csv"
COLLECTION_NAME = "tagged_summary_collection"
MODEL_NAME = "all-MiniLM-L6-v2"

# ---------- Load SentenceTransformer model ----------
model = SentenceTransformer(MODEL_NAME)

# ---------- Initialize in-memory ChromaDB client ----------
client = chromadb.Client(Settings(anonymized_telemetry=False))

# ---------- Create new collection (in-memory only) ----------
collection = client.get_or_create_collection(name=COLLECTION_NAME)

# ---------- Step: Read tagged summaries and insert into ChromaDB ----------
with open("tagged_summary.txt", "r", encoding="utf-8") as f:
    text = f.read()

entries = re.findall(r'"(.*?)"', text, re.DOTALL)
entries = entries[:500]  # Optional limit
documents = [entry.strip() for entry in entries]
embeddings = model.encode(documents, show_progress_bar=True)
ids = [str(uuid.uuid4()) for _ in documents]
metadatas = [{"source": "tagged_summary"} for _ in documents]

collection.add(
    documents=documents,
    embeddings=embeddings.tolist(),
    metadatas=metadatas,
    ids=ids
)

print("Documents loaded into Chroma (in-memory, no persistence)")

# ---------- Utility to render PDF first page ----------
def render_first_page(arxiv_id: str) -> str:
    try:
        prefix, num_version = arxiv_id.split("-")
        number = num_version.split("v")[0]
        url_id = f"{prefix}/{number}"
    except Exception as e:
        raise ValueError("Invalid arXiv ID format.") from e

    pdf_url = f"https://arxiv.org/pdf/{url_id}.pdf"

    temp_dir = tempfile.gettempdir()
    pdf_path = os.path.join(temp_dir, f"{arxiv_id}.pdf")
    img_path = os.path.join(temp_dir, f"{arxiv_id}_page1.png")

    if not os.path.exists(pdf_path):
        response = requests.get(pdf_url)
        response.raise_for_status()
        with open(pdf_path, "wb") as f:
            f.write(response.content)

    doc = fitz.open(pdf_path)
    page = doc.load_page(0)
    pix = page.get_pixmap(dpi=200)
    pix.save(img_path)
    return img_path

def get_recommendations(query: str, k: int = 5):
    if not query.strip():
        return None, "Please enter a search query.", None

    query_embedding = model.encode([query])
    results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=k
    )

    doc_ids = [doc.split()[0].strip() for doc in results['documents'][0]]

    df = pd.read_csv(CSV_PATH)
    matched_df = df[df["id"].isin(doc_ids)]

    if matched_df.empty:
        return None, "No matching papers found.", None

    # Ensure order of papers matches query result
    matched_df["id"] = pd.Categorical(matched_df["id"], categories=doc_ids, ordered=True)
    matched_df = matched_df.sort_values("id")

    arxiv_id = matched_df.iloc[0]['id']
    img_path = render_first_page(arxiv_id)
    category, rest = arxiv_id.split('-', 1)
    number = rest.split('v')[0]  
    arxiv_url = f"https://arxiv.org/abs/{category}/{number}"

    paper_titles = matched_df["title"].tolist()
    distances = results["distances"][0]
    similarity_scores = [1 - d for d in distances]

    # Ensure lengths match
    min_len = min(len(paper_titles), len(similarity_scores))
    paper_titles = paper_titles[:min_len]
    similarity_scores = similarity_scores[:min_len]

    # Save chart to working directory
    chart_path = "similarity_chart.png"
    plt.figure(figsize=(10, 4))
    plt.barh(paper_titles, similarity_scores, color='skyblue')
    plt.xlabel("Similarity Score (1 - Distance)")
    plt.title("Top-k Paper Similarities to Query")
    plt.tight_layout()
    plt.savefig(chart_path)
    plt.close()

    return img_path, f"[View paper on arXiv]({arxiv_url})", chart_path

gr.Interface(
    fn=get_recommendations,
    inputs=gr.Textbox(label="Search Query"),
    outputs=[
        gr.Image(label="First Page of PDF"),
        gr.Markdown(label="arXiv Paper Link"),
        gr.Image(label="Similarity Score Bar Chart")
    ],
    title="Semantic Paper Recommender",
    description="Enter a search query to find and preview scientific papers from arXiv."
).launch()