import streamlit as st
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import tempfile
import os

# Load env vars
load_dotenv()

st.set_page_config(page_title="RAG Chatbot with Gemini", layout="wide")
st.title("🧠 RAG Chatbot with Web URLs + Gemini")

# Gemini API key
gemini_key = st.text_input("🔑 Enter Gemini API Key", type="password")
os.environ["GOOGLE_API_KEY"] = gemini_key.strip()

# Web URLs input
urls_input = st.text_area("🌐 Enter Web URLs (comma-separated)")
load_docs_btn = st.button("🔍 Load Documents and Initialize Chat")

# Session state for vectorstore and chat history
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Function: Load and embed documents
def load_and_embed(urls):
    loader = WebBaseLoader(urls)
    docs = loader.load()
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        vectorstore = FAISS.from_documents(chunks, embeddings)
        vectorstore.save_local(tmpdir)
        vectorstore = FAISS.load_local(tmpdir, embeddings, allow_dangerous_deserialization=True)
        return vectorstore

# Load documents
if load_docs_btn and urls_input:
    urls = [url.strip() for url in urls_input.split(",") if url.strip()]
    with st.spinner("Processing documents..."):
        st.session_state.vectorstore = load_and_embed(urls)
    st.success("✅ Documents loaded and embedded. Start chatting below!")

# Chatbot interface
if st.session_state.vectorstore:
    user_question = st.text_input("💬 Ask a question")

    if user_question:
        retriever = st.session_state.vectorstore.as_retriever()
        docs = retriever.get_relevant_documents(user_question)
        context = "\n\n".join(doc.page_content for doc in docs[:3])

        # Append previous Q&A to context
        full_context = ""
        for qa in st.session_state.chat_history:
            full_context += f"Q: {qa['question']}\nA: {qa['answer']}\n\n"
        full_context += f"Context:\n{context}\n\nQ: {user_question}\nA:"

        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)


        response = llm.invoke(full_context)
        answer = response.content

        # Save in history
        st.session_state.chat_history.append({
            "question": user_question,
            "answer": answer
        })

        # Show full conversation
        for qa in reversed(st.session_state.chat_history):
            with st.chat_message("user"):
                st.markdown(qa["question"])
            with st.chat_message("assistant"):
                st.markdown(qa["answer"])
