# =======================================
# Block 1: Streamlit Setup + UI Skeleton
# =======================================
import os
import streamlit as st
import pandas as pd
import tempfile
import matplotlib.pyplot as plt
from datetime import datetime

# Config
CSV_PATH = "/home/ubuntu/final_project_capstone/data/sample.csv"
CHROMA_PATH = "/home/ubuntu/final_project_capstone/chroma_db"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Streamlit setup
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
        </style>
    """, unsafe_allow_html=True)

st.title("🤖 Amazon Product Review Chatbot")
st.markdown("Ask **product-related questions** and get **context-aware answers** from reviews and uploaded manuals.")

# PDF Upload
st.sidebar.header("📄 Upload Product Manual")
pdf_file = st.sidebar.file_uploader("Upload PDF", type=["pdf"])

# Question
st.subheader("💬 Ask a Product Question")
query = st.text_input("Type your question here...")

# Placeholder: future blocks
st.markdown("---")
st.write("**[Continue to next blocks for LLMs, embedding, analytics, etc.]**")
