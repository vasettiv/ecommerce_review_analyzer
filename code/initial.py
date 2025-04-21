from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
import chromadb
import pandas as pd
import fitz  # PyMuPDF
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# --- Step 1: Load and prepare review data ---
df = pd.read_csv("/home/ubuntu/final_project_capstone/data/sample.csv")

# Set this to your desired PDF file
DEFAULT_PDF_PATH = "/home/ubuntu/final_project_capstone/data/Amazon_Tap_Quick_Start_Guide.pdf"

documents = [
    (str(title).strip() + ". " + str(text).strip()).strip()
    for title, text in zip(df["reviews.title"], df["reviews.text"])
]

print(f"✅ Loaded {len(documents)} reviews from CSV.")

# --- Step 2: Generate embeddings ---
embedder = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embedder.encode(documents).tolist()

# --- Step 3: Store in ChromaDB ---
chroma_client = chromadb.Client(chromadb.config.Settings(
    persist_directory="/home/ubuntu/final_project_capstone/chroma_db",
    anonymized_telemetry=False
))

collection = chroma_client.get_or_create_collection(name="reviews")

collection.add(
    documents=documents,
    embeddings=embeddings,
    ids=[f"doc{i}" for i in range(len(documents))],
    metadatas=[{"source": "csv"} for _ in range(len(documents))]
)

print(f"📦 ChromaDB loaded with {collection.count()} documents from reviews.")

# --- Step 4: Load Mistral model (4-bit quant) ---
model_id = "mistralai/Mistral-7B-Instruct-v0.3"
bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype="float16")

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    quantization_config=bnb_config,
    trust_remote_code=True
)

generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

# --- PDF Extraction & Ingestion ---
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text.strip()

def add_pdf_to_chromadb(pdf_path):
    text = extract_text_from_pdf(pdf_path)
    chunks = [text[i:i+1000] for i in range(0, len(text), 1000)]

    embeddings = embedder.encode(chunks).tolist()
    doc_ids = [f"pdfdoc_{os.path.basename(pdf_path)}_{i}" for i in range(len(chunks))]

    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=doc_ids,
        metadatas=[{"source": "pdf", "filename": os.path.basename(pdf_path)} for _ in chunks]
    )
    print(f"📄 PDF '{os.path.basename(pdf_path)}' added with {len(chunks)} chunks.")

# --- Step 5: Question-answering function ---
def ask_question(query: str):
    query_embedding = embedder.encode([query]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=10)

    matched_docs = results['documents'][0]
    context = "\n".join(matched_docs)

    prompt = f"You are a helpful assistant answering questions based on the chat history and the following customer reviews and the document, answer the question and if you don't know you can say that you don't know but make sure that you dont hallucinate:\n\n{context}\n\nQuestion: {query}\nAnswer:"
    full_response = generator(prompt, max_new_tokens=300)[0]['generated_text']
    answer = full_response.split("Answer:")[-1].strip()
    return answer

# --- Step 6: Auto-load default PDF before chatbot starts ---
if os.path.exists(DEFAULT_PDF_PATH):
    add_pdf_to_chromadb(DEFAULT_PDF_PATH)
else:
    print(f"⚠️ PDF path '{DEFAULT_PDF_PATH}' not found. Skipping auto-upload.")

# --- Step 7: Chatbot loop with optional PDF upload ---
if __name__ == "__main__":
    print("\n💬 Chatbot is ready! Ask anything about the product reviews or uploaded PDF.")
    print("Type 'upload <path_to_pdf>' to add more product documents.")
    print("Type 'exit' or 'quit' to end the chat.\n")

    while True:
        user_input = input("❓ You: ").strip()

        if user_input.lower() in ['exit', 'quit']:
            print("👋 Chatbot session ended. Have a great day!")
            break

        # PDF Upload Command
        if user_input.lower().startswith("upload "):
            pdf_path = user_input[7:].strip()
            if os.path.exists(pdf_path) and pdf_path.lower().endswith(".pdf"):
                add_pdf_to_chromadb(pdf_path)
            else:
                print("⚠️ Invalid PDF path. Please check the file and try again.")
            print("-" * 60)
            continue

        # Ask question
        answer = ask_question(user_input)
        print("\n🤖 Bot:", answer)
        print("-" * 60)
