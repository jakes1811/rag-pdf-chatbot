# 📄 AI PDF Chatbot using RAG

An AI-powered PDF Question Answering system built using **LangChain**, **GROQ**, **ChromaDB**, and **Streamlit**.

Upload any PDF and chat with it using Retrieval-Augmented Generation (RAG).

---

## 🚀 Live Demo

🔗 Add your Streamlit app URL here

Example:
https://rag-pdf-chatbot-hty4yv5ivhwsaxvp2whwpx.streamlit.app/

---

## ✨ Features

- 📄 Upload PDF documents
- 🤖 Ask questions about the uploaded PDF
- 🔍 Semantic search using vector embeddings
- 🧠 Context-aware answers using RAG
- ⚡ Fast responses powered by GROQ Llama 3.1
- 💬 Interactive Streamlit chat interface
- 📌 Displays source chunks used for answers

---

## 🛠 Tech Stack

- Python
- Streamlit
- LangChain
- GROQ API
- ChromaDB
- HuggingFace Embeddings
- Sentence Transformers
- PyPDF

---

## 🧠 How It Works

1. User uploads a PDF
2. PDF text is extracted using PyPDFLoader
3. Text is split into chunks
4. Embeddings are generated using HuggingFace models
5. Chunks are stored in ChromaDB vector database
6. Relevant chunks are retrieved based on the question
7. GROQ Llama 3.1 generates context-aware answers

---

## 📂 Project Structure

```bash
rag-pdf-chatbot/
│
├── app.py
├── requirements.txt
├── runtime.txt
└── README.md
