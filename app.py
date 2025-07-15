"""
To address the user's request, I will modify the `app.py` file to:

1.  **Add a creativity bar (slider) for temperature**: Implement a Streamlit slider that allows users to select a value from 0 to 100%. This value will then be converted to a 0.0-1.0 range for the `temperature` parameter in the `ChatGoogleGenerativeAI` model.
2.  **Adjust URL input for new lines**: Change the `st.text_area` for URLs to split input by newlines instead of commas.

Here's the updated `app.py` code:
"""
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

# Creativity bar for temperature
creativity_percent = st.slider("✨ Creativity (Temperature)", 0, 100, 0)
temperature_value = creativity_percent / 100.0

# URL input - now split by new lines
urls_input = st.text_area("🌐 Enter Web URLs (one per line)")
load_docs_btn = st.button("🔍 Load Documents and Initialize Chat")

# Session state for vectorstore and chat history
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "temperature" not in st.session_state:
    st.session_state.temperature = 0.0 # Default value

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
    urls = [url.strip() for url in urls_input.split("\n") if url.strip()] # Split by new line
    st.session_state.temperature = temperature_value # Store temperature in session state
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

        # Use the temperature from session state
        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=st.session_state.temperature)
        response = llm.invoke(prompt)
        answer = response.content.strip()

        # Save the Q&A to history
        st.session_state.chat_history.append({  # Corrected from st.session_session_state
            "question": user_question,
            "answer": answer
        })

        # Display chat history
        for qa in reversed(st.session_state.chat_history):
            with st.chat_message("user"):
                st.markdown(qa["question"])
            with st.chat_message("assistant"):
                st.markdown(qa["answer"])
