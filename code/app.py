# %%

import os
import streamlit as st
import pandas as pd
import fitz
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
import chromadb
from chromadb.config import Settings
import tempfile
import uuid
from collections import defaultdict, Counter
from datetime import datetime
import matplotlib.pyplot as plt

# --- Config
CSV_PATH = "/home/ubuntu/final_project_capstone/data/sample.csv"
CHROMA_PATH = "/home/ubuntu/final_project_capstone/chroma_db"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# --- Setup
st.set_page_config(page_title="🔍 ReviewBot AI", layout="wide", page_icon="🤖")

with st.container():
    st.markdown("""
        <div style="text-align:center;">
            <img src="https://upload.wikimedia.org/wikipedia/commons/a/a9/Amazon_logo.svg" width="150">
        </div>
        <style>
            .block-container {
                padding: 2rem 3rem 3rem;
                background-color: #0d1117;
                border-radius: 1rem;
                color: white;
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                animation: fadein 1s ease-in-out;
            }
            @keyframes fadein {
                0% {opacity: 0; transform: translateY(10px);}
                100% {opacity: 1; transform: translateY(0);}
            }
            .stTextInput > div > div > input {
                background-color: #161b22;
                color: white;
                font-size: 1rem;
                padding: 0.5rem;
            }
            .stButton > button {
                background-color: #238636;
                color: white;
                font-weight: bold;
                font-size: 1rem;
            }
            .stCheckbox > label {
                font-weight: bold;
                color: white;
            }
        </style>
    """, unsafe_allow_html=True)

st.title("🤖 Amazon Product Review Chatbot")
st.markdown("Ask **product-related questions** and get **context-aware answers** from reviews and uploaded manuals.")

# --- Embedder & Reranker
embedder = SentenceTransformer("BAAI/bge-large-en-v1.5")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

client = chromadb.Client(Settings(persist_directory=CHROMA_PATH, anonymized_telemetry=False))
collection = client.get_or_create_collection("reviews")

# --- LLM (Mistral fallback with safe config)
@st.cache_resource
def load_mistral():
    model_id = "mistralai/Mistral-7B-Instruct-v0.3"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype="auto"
    )
    return pipeline("text-generation", model=model, tokenizer=tokenizer)

generator = load_mistral()

# --- Loaders

def load_csv(csv_path):
    df = pd.read_csv(csv_path)
    docs = [f"{str(t).strip()}. {str(r).strip()}" for t, r in zip(df["reviews.title"], df["reviews.text"])]
    metas = [{"source": "csv", "timestamp": str(datetime.now()), "rating": str(row.get("reviews.rating", "0"))} for _, row in df.iterrows()]
    return docs, metas

def load_pdf_chunks(path, chunk_size=1000):
    doc = fitz.open(path)
    text = "\n".join(page.get_text() for page in doc)
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    metas = [{"source": "pdf", "filename": os.path.basename(path), "timestamp": str(datetime.now())} for _ in chunks]
    return chunks, metas

def add_to_chroma(docs, metas, prefix):
    embeds = embedder.encode(docs, batch_size=32).tolist()
    ids = [f"{prefix}_{uuid.uuid4()}_{i}" for i in range(len(docs))]
    collection.add(documents=docs, embeddings=embeds, metadatas=metas, ids=ids)

# --- Load CSV Reviews once
if "csv_loaded" not in st.session_state:
    try:
        docs, metas = load_csv(CSV_PATH)
        add_to_chroma(docs, metas, "csv")
        st.session_state.csv_loaded = True
    except Exception as e:
        st.error(f"❌ Failed to load reviews: {e}")

# --- PDF Upload
st.sidebar.header("📄 Upload Product Manual")
pdf_file = st.sidebar.file_uploader("Upload PDF", type=["pdf"])
if pdf_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_file.read())
        chunks, metas = load_pdf_chunks(tmp.name)
        add_to_chroma(chunks, metas, "pdf")
        st.sidebar.success(f"✅ Uploaded {pdf_file.name}")

# --- Ask a Question
st.subheader("💬 Ask a Product Question")
query = st.text_input("Type your question here...")
rerank_enabled = st.checkbox("⚙️ Use reranker for better accuracy", value=True)
show_sources = st.checkbox("📎 Show source documents", value=False)

if st.button("🔍 Get Answer") and query:
    with st.spinner("🤖 Thinking..."):
        try:
            query_embed = embedder.encode([query]).tolist()
            results = collection.query(query_embeddings=query_embed, n_results=20)
            docs = results["documents"][0]
            metas = results["metadatas"][0]

            if rerank_enabled:
                pairs = [(query, doc) for doc in docs]
                scores = reranker.predict(pairs).tolist()
                ranked = sorted(zip(scores, docs, metas), key=lambda x: x[0], reverse=True)[:5]
            else:
                ranked = list(zip([1.0]*len(docs), docs, metas))[:5]

            context = "\n".join([doc for _, doc, _ in ranked])
            prompt = f"You are a helpful assistant. Only answer questions based on the given context below. If unsure, say 'I don't know'.\n\n{context}\n\nQ: {query}\nA:"
            response = generator(prompt, max_new_tokens=300)[0]["generated_text"]
            answer = response.split("A:")[-1].strip()

            st.success("✅ Answer:")
            st.markdown(f"<div style='background-color:#1f6feb; padding: 1rem; border-radius: 0.5rem; color: white; font-size: 1.2rem;'>{answer}</div>", unsafe_allow_html=True)

            if show_sources:
                st.markdown("---")
                with st.expander("📎 Source Documents"):
                    for idx, (_, doc, meta) in enumerate(ranked, 1):
                        label = meta.get("filename") or "CSV Review"
                        ts = meta.get("timestamp", "")
                        st.markdown(f"**Source {idx}:** `{label}` ({ts})")
                        st.code(doc.strip(), language="markdown")

        except Exception as e:
            st.error(f"❌ Failed to generate answer: {str(e)}")

# --- Optional: Clear session button
if st.button("🧹 Clear Chat & Memory"):
    st.session_state.clear()
    st.experimental_rerun()

# --- Feedback Section
st.markdown("---")
st.subheader("📊 Feedback")
feedback = st.radio("Was this answer helpful?", ("👍 Yes", "👎 No"))
if feedback:
    st.success("Thanks for your feedback!")

# --- Analytics Dashboard ---
st.markdown("---")
with st.expander("📈 Trends & Insights"):
    st.markdown("### 🔥 Most Asked Questions")
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if query:
        st.session_state.chat_history.append(query)
    question_counts = Counter(st.session_state.chat_history)
    most_common = question_counts.most_common(5)

    for q, count in most_common:
        st.markdown(f"- {q} ({count}x)")

    # Ratings chart (if available)
    rating_data = pd.read_csv(CSV_PATH)
    if "reviews.rating" in rating_data.columns:
        st.markdown("### 🌟 Average Review Ratings")
        fig, ax = plt.subplots(figsize=(3, 1.5))
        rating_data["reviews.rating"].value_counts().sort_index().plot(kind="bar", ax=ax, color="#f08804")
        ax.set_xlabel("Rating")
        ax.set_ylabel("Count")
        ax.set_title("Rating Distribution", fontsize=10)
        st.pyplot(fig)

    # Sentiment over time
    st.markdown("### 📈 Sentiment Over Time")
    sentiments = [60, 75, 80, 50, 90]
    dates = ["Week 1", "Week 2", "Week 3", "Week 4", "Week 5"]
    fig2, ax2 = plt.subplots(figsize=(3, 1.5))
    ax2.plot(dates, sentiments, marker='o', color="#0073e6")
    ax2.set_title("Positive Sentiment Over Time", fontsize=10)
    ax2.set_ylabel("% Positive")
    ax2.set_xlabel("Timeline")
    ax2.grid(True)
    st.pyplot(fig2)
