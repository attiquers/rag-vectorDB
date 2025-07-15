"""
This Streamlit application provides a RAG Chatbot that can leverage both
Google Gemini and OpenAI models for answering questions based on provided
web URLs. Users can input API keys for both services and select their preferred
model.

This updated version now implements a two-step process for answer generation:
1. An initial answer is generated from the document context and chat history.
2. This initial answer is then re-evaluated by the LLM against the same context
   and history to check for relevance and refine it, ensuring no external
   knowledge is used, but allowing for creative synthesis within the provided information.
   The relevance check in the second step is now more lenient, allowing for answers
   that are 'vaguely similar' or 'at least 80% relevant' to the context.
   The creativity (temperature) control has been removed.
   User input is now submitted explicitly via a send button or Enter key.
   A loading icon is shown while the AI is thinking, replacing the answer area temporarily.
   The chat history is now displayed below the chat input form.
"""
import streamlit as st
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import tempfile
import os

# Load environment variables
load_dotenv()

st.set_page_config(page_title="🧠 Langchain Chatbot", layout="wide")
st.title("🧠 Langchain Chatbot")

# API key inputs
gemini_key = st.text_input("🔑 Enter Gemini API Key", type="password")
openai_key = st.text_input("🔑 Enter OpenAI API Key", type="password")

# Model selection
model_choice = st.radio(
    "🤖 Choose your AI Model",
    ("Gemini", "OpenAI"),
    index=0
)

# Set API keys based on choice (only for initialization, actual usage will check if key is provided)
if model_choice == "Gemini":
    os.environ["GOOGLE_API_KEY"] = gemini_key.strip()
    selected_model_name = "gemini-2.0-flash"
else: # OpenAI
    os.environ["OPENAI_API_KEY"] = openai_key.strip()
    # gpt-4o-mini is one of the least expensive OpenAI models
    selected_model_name = "gpt-4o-mini" 

# URL input - now split by new lines
urls_input = st.text_area("🌐 Enter Web URLs (one per line)")
load_docs_btn = st.button("🔍 Load Documents and Initialize Chat")

# Session state for vectorstore and chat history
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Ensure chat history has the correct structure for existing (possibly old) entries
# This helps prevent KeyError if old session state is present from previous runs
# A more robust solution for mixed types would involve clearing or full conversion.
# For simplicity, if an old 'question'/'answer' entry is found, we convert it for the current run.
cleaned_history = []
for message in st.session_state.chat_history:
    if "question" in message and "answer" in message and "role" not in message:
        cleaned_history.append({"role": "user", "content": message["question"]})
        cleaned_history.append({"role": "assistant", "content": message["answer"]})
    else:
        cleaned_history.append(message)
st.session_state.chat_history = cleaned_history


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
    urls = [url.strip() for url in urls_input.split("\n") if url.strip()]
    with st.spinner("Processing documents..."):
        st.session_state.vectorstore = load_and_embed(urls)
    st.success("✅ Documents loaded and embedded. Start chatting below!")

# Chatbot section
if st.session_state.vectorstore:
    # Use st.form for explicit submission
    with st.form("chat_form", clear_on_submit=True):
        user_question = st.text_input("💬 Ask a question", key="user_question_input")
        submitted = st.form_submit_button("Send ✈️")

        if submitted and user_question:
            # Build history for prompt from *existing* messages (prior turns)
            prior_chat_history_for_prompt = ""
            for msg in st.session_state.chat_history:
                prior_chat_history_for_prompt += f"{msg['role'].capitalize()}: {msg['content']}\n"

            # Add user message to chat history immediately for display
            st.session_state.chat_history.append({"role": "user", "content": user_question})

            # Display user message immediately in the chat area
            with st.chat_message("user"):
                st.markdown(user_question)

            # Use st.status for showing thinking process and the response
            with st.chat_message("assistant"):
                with st.status("Thinking...", expanded=True) as status:
                    retriever = st.session_state.vectorstore.as_retriever()
                    docs = retriever.get_relevant_documents(user_question)
                    document_context = "\n\n".join(doc.page_content for doc in docs[:3])
                    
                    llm = None
                    fixed_temperature = 0.0 
                    if model_choice == "Gemini":
                        if gemini_key:
                            llm = ChatGoogleGenerativeAI(model=selected_model_name, temperature=fixed_temperature)
                        else:
                            answer = "Please enter your Gemini API Key to use Gemini models." # Fallback answer
                    elif model_choice == "OpenAI":
                        if openai_key:
                            llm = ChatOpenAI(model=selected_model_name, temperature=fixed_temperature)
                        else:
                            answer = "Please enter your OpenAI API Key to use OpenAI models." # Fallback answer
                    
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
                            chat_history=prior_chat_history_for_prompt, # Use prior history
                            question=user_question
                        )
                        initial_response = llm.invoke(initial_prompt)
                        initial_answer = initial_response.content.strip()

                        # Step 2: Relevance check and refinement
                        relevance_check_prompt_template = PromptTemplate.from_template("""
You are a validator AI. Your task is to review an initial answer for a question, using only the provided document context and chat history.
DO NOT use any external knowledge.

**Relevance Criteria:**
- If the initial answer is directly supported by the document context and chat history, provide the refined answer.
- If the initial answer is not directly present but is *at least 80% relevant* or vaguely similar to the information in the document context and chat history, you should still consider it relevant and provide the refined answer.
- If the initial answer is genuinely not supported or cannot be inferred from the provided document context and chat history, then and *only then* should you state: "This information is not in the URLs pages provided or previous conversation."

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
                            chat_history=prior_chat_history_for_prompt, # Use prior history
                            original_question=user_question,
                            initial_answer=initial_answer
                        )
                        final_response = llm.invoke(final_prompt)
                        answer = final_response.content.strip()
                    else:
                        answer = "Please provide valid API keys." # Fallback if LLM not initialized
                    
                    status.update(label="Response generated!", state="complete", expanded=False)
                
                st.markdown(answer) # Display the final answer after status complete
                
                # Append the assistant's final message to chat history for next turn's display
                st.session_state.chat_history.append({"role": "assistant", "content": answer})
                st.rerun() # Rerun to update the entire chat history and clear the form input

    # Display chat messages from history AFTER the input form
    # This loop will ensure all previous and current messages are displayed correctly
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
