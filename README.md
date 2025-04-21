# 💬 Product Review & PDF Chatbot with RAG and Streamlit

This project implements an interactive chatbot that provides answers based on customer product reviews and uploaded PDF documentation. It leverages advanced AI techniques, including **Retrieval-Augmented Generation (RAG)** using **Mistral-7B-Instruct-v0.3**, embeddings with **SentenceTransformers**, and document storage with **ChromaDB**.

---

## 🚀 Features

- **Interactive Chatbot:** Ask questions based on product reviews or PDF documents.
- **PDF Support:** Easily ingest and search through PDF documents.
- **RAG-Based Answering:** Ensures accurate answers with reduced hallucinations.
- **Quantized LLM (Mistral-7B):** Optimized for efficient inference on GPU.

---

## 🛠️ Technology Stack

- **Python:** Programming language.
- **Streamlit:** Interactive web-based user interface.
- **ChromaDB:** Vector database for efficient retrieval of embeddings.
- **SentenceTransformers:** Generates high-quality embeddings.
- **Hugging Face Transformers & BitsAndBytes:** To run quantized LLM efficiently.
- **PyMuPDF (fitz):** Extract text from PDFs.

---


## ⚙️ Installation

### Step 1: Clone the repository
```bash
git clone <your_repository_url>
cd final_project_capstone
```
### Step 2: Create a virtual environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install dependencies
```bash
pip install streamlit sentence-transformers transformers bitsandbytes chromadb pymupdf torch torchvision torchaudio

```

### Step 4: Ensure CUDA is available (optional but recommended)

```bash
nvidia-smi
```

🚦 Quick Start

### Run the Streamlit chatbot app
```bash
streamlit run app.py --server.runOnSave false
```
Open your browser and go to [http://localhost:8501](http://localhost:8501).

---

## 📝 How to Use

- **Chatting:** Type questions directly into the Streamlit interface.
- **Uploading PDFs:** Use the built-in PDF upload feature to add more product documentation dynamically.
- **Exit Chat:** Simply stop the Streamlit server (`Ctrl+C`) in your terminal.

---

## 🖥️ GPU Requirements

This setup is tested on GPU environments (e.g., **NVIDIA A10G with ≥16GB VRAM**).

If you face GPU memory issues, modify the model load configuration in `app.py`:

```python
device_map="auto"  # Use device_map={"": "cpu"} to force CPU-only mode

```
---

## 🛡️ Handling Errors

### Common Issues:

- **GPU Memory Issues**
  - Reduce `max_new_tokens` in the `generator()` call.
  - Use CPU fallback mode by modifying your model loading in `app.py`:

    ```python
    device_map = {"": "cpu"}
    ```

- **Streamlit Watcher Errors** (safe to ignore):

    ```bash
    streamlit run app.py --server.runOnSave false
    ```

---

## 📚 References & Resources

- [Streamlit Documentation](https://docs.streamlit.io/)
- [ChromaDB GitHub](https://github.com/chroma-core/chroma)
- [Mistral-7B Model Card](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3)
- [SentenceTransformers GitHub](https://github.com/UKPLab/sentence-transformers)

---

## 🤝 Contributing

Feel free to open issues, submit pull requests, or reach out with suggestions or feedback!
