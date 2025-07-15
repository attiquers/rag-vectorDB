"""
This Streamlit application provides a RAG Chatbot that can leverage both
Google Gemini and OpenAI models for answering questions based on provided
web URLs. Users can input API keys for both services, select their preferred
model, and adjust the creativity (temperature) of the AI.

This updated version now implements a two-step process for answer generation:
1. An initial answer is generated from the document context and chat history.
2. This initial answer is then re-evaluated by the LLM against the same context
   and history to check for relevance and refine it, ensuring no external
   knowledge is used, but allowing for creative synthesis within the provided information.
"""
import streamlit as st
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI # Import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import tempfile
import os

# Load environment variables
load_dotenv()

st.set_page_config(page_title="RAG Chatbot with Web URLs + Gemini/OpenAI", layout="wide")
st.title("🧠 RAG Chatbot with Web URLs + Gemini/OpenAI")

# API key inputs
gemini_key = st.text_input("🔑 Enter Gemini API Key", type="password")
openai_key = st.text_input("🔑 Enter OpenAI API Key", type="password") # New OpenAI key input

# Model selection
model_choice = st.radio(
    "🤖 Choose your AI Model",
    ("Gemini", "OpenAI"),
    index=0 # Default to Gemini
)

# Set API keys based on choice (only for initialization, actual usage will check if key is provided)
if model_choice == "Gemini":
    os.environ["GOOGLE_API_KEY"] = gemini_key.strip()
    selected_model_name = "gemini-2.0-flash" # Default Gemini model
else: # OpenAI
    os.environ["OPENAI_API_KEY"] = openai_key.strip()
    # gpt-4o-mini is one of the least expensive OpenAI models
    selected_model_name = "gpt-4o-mini" 

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
        document_context = "\n\n".join(doc.page_content for doc in docs[:3])

        # Format chat history for inclusion in the prompt
        history_for_prompt = ""
        for qa in st.session_state.chat_history:
            history_for_prompt += f"User: {qa['question']}\nAssistant: {qa['answer']}\n"
        
        # Initialize LLM based on user's choice and provided key
        llm = None
        if model_choice == "Gemini":
            if gemini_key:
                llm = ChatGoogleGenerativeAI(model=selected_model_name, temperature=st.session_state.temperature)
            else:
                st.warning("Please enter your Gemini API Key to use Gemini models.")
        elif model_choice == "OpenAI":
            if openai_key:
                llm = ChatOpenAI(model=selected_model_name, temperature=st.session_state.temperature)
            else:
                st.warning("Please enter your OpenAI API Key to use OpenAI models.")
        
        if llm:
            # Step 1: Generate initial answer from context and history
            initial_answer_prompt_template = PromptTemplate.from_template("""
You are an intelligent assistant. Based on the provided document context and chat history, answer the following question.
Focus on providing a comprehensive answer using *only* the information available in the context and history.
If the information is not present, you may state that, but try to infer or synthesize from what is given if possible.

Document Context:
{document_context}

Chat History:
{chat_history}

Question:
{question}

Answer:""")
            
            initial_prompt = initial_answer_prompt_template.format(
                document_context=document_context,
                chat_history=history_for_prompt,
                question=user_question
            )
            initial_response = llm.invoke(initial_prompt)
            initial_answer = initial_response.content.strip()

            # Step 2: Relevance check and refinement
            relevance_check_prompt_template = PromptTemplate.from_template("""
You are a validator AI. Your task is to review an initial answer for a question, using only the provided document context and chat history.
DO NOT use any external knowledge.
If the initial answer is well-supported by the document context and chat history, or can be improved using *only* that information, provide the refined answer.
If the initial answer is *not* sufficiently supported by the provided document context and chat history, or if the question cannot be answered from the provided information, state: "This information is not in the URLs pages provided or previous conversation."

Document Context:
{document_context}

Chat History:
{chat_history}

Original Question:
{original_question}

Initial Answer to Validate:
{initial_answer}

Refined Answer (or "This information is not in the URLs pages provided or previous conversation."):""")

            final_prompt = relevance_check_prompt_template.format(
                document_context=document_context,
                chat_history=history_for_prompt,
                original_question=user_question,
                initial_answer=initial_answer
            )
            final_response = llm.invoke(final_prompt)
            answer = final_response.content.strip()

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
