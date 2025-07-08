import streamlit as st
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import tempfile
import os

# Load environment variables
load_dotenv()

st.set_page_config(page_title="RAG Chatbot with Gemini", layout="wide")
st.title("🧠 RAG Chatbot with Web URLs + Gemini")

# Gemini API key input
gemini_key = st.text_input("🔑 Enter Gemini API Key", type="password")
os.environ["GOOGLE_API_KEY"] = gemini_key.strip()

# URL input
urls_input = st.text_area("🌐 Enter Web URLs (comma-separated)")
load_docs_btn = st.button("🔍 Load Documents and Initialize Chat")

# Session state for vectorstore and chat history
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Function to load and embed documents
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

# Load and embed if triggered
if load_docs_btn and urls_input:
    urls = [url.strip() for url in urls_input.split(",") if url.strip()]
    with st.spinner("Processing documents..."):
        st.session_state.vectorstore = load_and_embed(urls)
    st.success("✅ Documents loaded and embedded. Start chatting below!")

# Chatbot section
if st.session_state.vectorstore:
    user_question = st.text_input("💬 Ask a question")

    if user_question:
        retriever = st.session_state.vectorstore.as_retriever()
        docs = retriever.get_relevant_documents(user_question)
        context = "\n\n".join(doc.page_content for doc in docs[:3])

        # Define strict context-only prompt
        prompt_template = PromptTemplate.from_template("""
You are an intelligent assistant. Use only the information from the provided context to answer the question.
If the answer is not found in the context, reply with: "This information is not in the URLs pages provided."

Context:
{context}

Question:
{question}

Answer:""")

        # Fill in the template
        prompt = prompt_template.format(context=context, question=user_question)

        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
        response = llm.invoke(prompt)
        answer = response.content.strip()

        # Save the Q&A to history
        st.session_state.chat_history.append({
            "question": user_question,
            "answer": answer
        })

        # Display chat history
        for qa in reversed(st.session_state.chat_history):
            with st.chat_message("user"):
                st.markdown(qa["question"])
            with st.chat_message("assistant"):
                st.markdown(qa["answer"])
