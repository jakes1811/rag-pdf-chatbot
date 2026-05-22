
import streamlit as st
import os
import tempfile

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq

st.set_page_config(
    page_title="PDF Chatbot",
    layout="wide"
)

st.title("📄 AI PDF Chatbot")
st.caption("Chat with your PDF using RAG + GROQ")

# Sidebar
groq_key = st.sidebar.text_input(
    "🔑 Enter GROQ API Key",
    type="password"
)

if groq_key:
    os.environ["GROQ_API_KEY"] = groq_key

uploaded_file = st.sidebar.file_uploader(
    "📁 Upload PDF",
    type="pdf"
)

# Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

if "retriever" not in st.session_state:
    st.session_state.retriever = None

if "llm" not in st.session_state:
    st.session_state.llm = None


# Process PDF
if uploaded_file and groq_key:

    if st.sidebar.button("⚡ Process PDF"):

        with st.spinner("Processing PDF..."):

            # Save temp file
            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".pdf"
            ) as f:

                f.write(uploaded_file.read())

                pdf_path = f.name

            # Load PDF
            loader = PyPDFLoader(pdf_path)

            docs = loader.load()

            # Split text
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )

            chunks = splitter.split_documents(docs)

            # Embeddings
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )

            # Vector DB
            ids = [str(i) for i in range(len(chunks))]
            vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                ids=ids
            )

            # Retriever
            retriever = vectorstore.as_retriever(
                search_kwargs={"k": 3}
            )

            # GROQ LLM
            llm = ChatGroq(
                model_name="llama-3.1-8b-instant"
            )

            st.session_state.retriever = retriever
            st.session_state.llm = llm

        st.sidebar.success(
            f"✅ PDF processed! {len(chunks)} chunks created."
        )

# Show chat history
for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):

        st.write(msg["content"])

# Chat input
question = st.chat_input(
    "Ask a question about your PDF..."
)

if question:

    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):

        st.write(question)

    if st.session_state.retriever:

        with st.chat_message("assistant"):

            with st.spinner("Thinking..."):

                docs = st.session_state.retriever.invoke(question)

                context = "\n\n".join([
                    doc.page_content for doc in docs
                ])

                prompt = f"""
Use the context below to answer the question.

If answer is not found in context,
say:
'I could not find it in the document.'

Context:
{context}

Question:
{question}

Answer:
"""

                response = st.session_state.llm.invoke(prompt)

                answer = response.content

                st.write(answer)

                with st.expander("📌 Source Chunks"):

                    for i, doc in enumerate(docs, 1):

                        st.markdown(
                            f"**Chunk {i}:** "
                            f"{doc.page_content[:300]}..."
                        )

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer
        })

    else:

        st.warning(
            "⚠️ Upload and process PDF first."
        )
