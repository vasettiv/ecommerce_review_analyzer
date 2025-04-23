import streamlit as st
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
import chromadb
from chromadb.config import Settings
import pandas as pd
import fitz
import os
from pathlib import Path
import uuid
import tempfile

# ---------------------- Config ----------------------
os.environ["TOKENIZERS_PARALLELISM"] = "false"
BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = os.getenv("CSV_PATH", str((BASE_DIR / "../data/sample.csv").resolve()))
DEFAULT_PDF_PATH = os.getenv("DEFAULT_PDF_PATH", str((BASE_DIR / "../data/Amazon_Tap_Quick_Start_Guide.pdf").resolve()))
CHROMA_PATH = os.getenv("CHROMA_PATH", str((BASE_DIR / "../chroma_db").resolve()))

# ---------------------- UI Setup ----------------------
st.set_page_config(page_title="ReviewBot AI", layout="wide", page_icon="🤖")
st.markdown("""
    <style>
        .block-container { background-color: #0d1117; color: white; padding: 2rem 3rem 3rem; border-radius: 1rem; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
        .stTextInput > div > div > input { background-color: #161b22; color: white; }
        .stButton > button { background-color: #238636; color: white; font-weight: bold; }
        .stCheckbox > label { font-weight: bold; color: white; }
    </style>
""", unsafe_allow_html=True)

st.title("🤖 Product QA and Review Chatbot")
st.markdown("Ask **product-related questions** and get **context-aware answers** from reviews and manuals.")

# ---------------------- Loaders ----------------------
@st.cache_resource
def load_reviews():
    df = pd.read_csv(CSV_PATH)
    return [f"{str(title).strip()}. {str(text).strip()}" for title, text in zip(df["reviews.title"], df["reviews.text"])]

@st.cache_resource
def get_embedder():
    return SentenceTransformer("BAAI/bge-large-en-v1.5")

@st.cache_resource
def get_reranker():
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

@st.cache_resource
def get_generator():
    model_id = "mistralai/Mistral-7B-Instruct-v0.3"
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype="float16")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, device_map="auto", quantization_config=bnb_config, trust_remote_code=True
    )
    return pipeline("text-generation", model=model, tokenizer=tokenizer)

@st.cache_resource
def get_chroma_collection(documents):
    embedder = get_embedder()
    client = chromadb.Client(Settings(persist_directory=CHROMA_PATH, anonymized_telemetry=False))
    collection = client.get_or_create_collection(name="reviews")
    if collection.count() == 0:
        embeddings = embedder.encode(documents).tolist()
        collection.add(
            documents=documents,
            embeddings=embeddings,
            ids=[f"doc{i}" for i in range(len(documents))],
            metadatas=[{"source": "csv"} for _ in documents]
        )
    return collection

# ---------------------- PDF & CSV Helpers ----------------------
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    return "\n".join(page.get_text() for page in doc).strip()

def add_pdf_to_chromadb(collection, pdf_path):
    embedder = get_embedder()
    text = extract_text_from_pdf(pdf_path)
    chunks = [text[i:i + 1000] for i in range(0, len(text), 1000)]
    embeddings = embedder.encode(chunks).tolist()
    doc_ids = [f"pdfdoc_{Path(pdf_path).stem}_{uuid.uuid4()}_{i}" for i in range(len(chunks))]
    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=doc_ids,
        metadatas=[{"source": "pdf", "filename": Path(pdf_path).name} for _ in chunks]
    )
    return len(chunks)

# ---------------------- QA Function ----------------------
def ask_question(query, history, collection, rerank=True):
    embedder = get_embedder()
    reranker = get_reranker()
    generator = get_generator()

    query_embedding = embedder.encode([query]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=30)
    docs = results["documents"][0]
    metas = results["metadatas"][0]

    if rerank:
        scores = reranker.predict([(query, doc) for doc in docs])
        ranked = sorted(zip(scores, docs, metas), reverse=True)[:10]
    else:
        ranked = [(1.0, doc, meta) for doc, meta in zip(docs, metas)][:10]

    context = "\n".join([doc for _, doc, _ in ranked])
    history_text = "\n".join([f"Q: {q}\nA: {a}" for q, a in history])
    prompt = (
        f"You are a helpful assistant answering product-related questions based on the provided documents. "
        f"Use the previous chat history and the following context to answer the current question. "
        f"Answer truthfully. If unsure, say 'I don't know'.\n\n"
        f"Chat History:\n{history_text}\n\n"
        f"Context:\n{context}\n\n"
        f"User: {query}\nAssistant:"
    )
    response = generator(prompt, max_new_tokens=300)[0]["generated_text"]
    return response.split("Assistant:")[-1].strip(), ranked

# ---------------------- Load Default Data ----------------------
documents = load_reviews()
collection = get_chroma_collection(documents)

if "default_pdf_loaded" not in st.session_state:
    try:
        added_chunks = add_pdf_to_chromadb(collection, DEFAULT_PDF_PATH)
        st.session_state.default_pdf_loaded = True
        st.sidebar.info(f"📄 Default manual loaded with {added_chunks} chunks.")
    except Exception as e:
        st.sidebar.error(f"❌ Failed to load default PDF: {e}")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ---------------------- Sidebar: Upload PDF (Append) ----------------------
with st.sidebar:
    st.header("📄 Upload Extra PDF Manual")
    uploaded_file = st.file_uploader("Upload PDF", type="pdf")
    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            count = add_pdf_to_chromadb(collection, str(tmp.name))
            st.success(f"✅ Added {count} chunks from {uploaded_file.name}")
            os.remove(tmp.name)

# ---------------------- Sidebar: Upload New Product (Replace) ----------------------
st.sidebar.markdown("### 🔄 Switch to a Different Product")
with st.sidebar.form("upload_new_product"):
    uploaded_csv = st.file_uploader("Upload New Product Reviews (CSV)", type=["csv"], key="csv_upload")
    uploaded_pdf = st.file_uploader("Upload Product Manual (PDF)", type=["pdf"], key="pdf_upload")
    submit_product = st.form_submit_button("Upload New Product")

    if submit_product:
        try:
            # Reset ChromaDB collection
            client = chromadb.Client(Settings(persist_directory=CHROMA_PATH, anonymized_telemetry=False))
            client.delete_collection("reviews")
            collection = client.get_or_create_collection(name="reviews")

            embedder = get_embedder()

            # Process CSV
            if uploaded_csv:
                df = pd.read_csv(uploaded_csv)
                if "reviews.title" in df.columns and "reviews.text" in df.columns:
                    review_docs = [f"{str(title).strip()}. {str(text).strip()}" for title, text in zip(df["reviews.title"], df["reviews.text"])]
                    embeddings = embedder.encode(review_docs, batch_size=32).tolist()
                    collection.add(
                        documents=review_docs,
                        embeddings=embeddings,
                        ids=[f"usercsv_{uuid.uuid4()}_{i}" for i in range(len(review_docs))],
                        metadatas=[{"source": "user_csv"} for _ in review_docs]
                    )
                    st.sidebar.success(f"✅ Uploaded and indexed {len(review_docs)} reviews.")
                else:
                    st.sidebar.error("❌ CSV must contain 'reviews.title' and 'reviews.text' columns.")

            # Process PDF
            if uploaded_pdf:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_pdf.read())
                    added_chunks = add_pdf_to_chromadb(collection, tmp.name)
                    st.sidebar.success(f"✅ Uploaded and indexed {added_chunks} chunks from PDF.")
                    os.remove(tmp.name)

            st.session_state.chat_history = []
            st.sidebar.info("🆕 Product context switched successfully!")
            st.experimental_rerun()

        except Exception as e:
            st.sidebar.error(f"❌ Failed to switch product: {e}")

# ---------------------- Chat UI ----------------------
st.subheader("💬 Ask a Product Question")
query = st.text_input("Your question:")
rerank = st.checkbox("⚙️ Use reranker", value=True)
show_sources = st.checkbox("📎 Show source documents", value=False)

if st.button("Ask") and query:
    with st.spinner("Generating answer..."):
        answer, ranked_docs = ask_question(query, st.session_state.chat_history, collection, rerank)
        st.session_state.chat_history.append((query, answer))
        if len(st.session_state.chat_history) > 5:
            st.session_state.chat_history.pop(0)

        st.success("✅ Answer:")
        st.markdown(f"**Answer:** {answer}")

        if show_sources:
            with st.expander("📎 Source Documents"):
                for i, (_, doc, meta) in enumerate(ranked_docs, 1):
                    label = meta.get("filename") or "CSV Review"
                    st.markdown(f"**Source {i}:** `{label}`")
                    st.code(doc.strip(), language="markdown")

# ---------------------- Clear Chat ----------------------
if st.button("🧹 Clear Chat"):
    st.session_state.chat_history = []
    st.experimental_rerun()

# ---------------------- Chat History Display ----------------------
if st.session_state.chat_history:
    st.markdown("---")
    st.subheader("🗨️ Chat History")
    for q, a in st.session_state.chat_history:
        st.markdown(f"**You:** {q}")
        st.markdown(f"**Bot:** {a}")
