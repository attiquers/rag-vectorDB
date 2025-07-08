import streamlit as st
import requests
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
import tempfile

# ---------- UI CONFIG ----------
st.set_page_config(page_title="Gemini-Powered RAG", layout="centered")
st.title("🔍 Gemini-Powered RAG from Web URLs")

with st.form("input_form"):
    gemini_key = st.text_input("🔐 Your Gemini API Key", type="password")
    urls_input = st.text_area("🌐 URLs to search from (one per line)")
    user_question = st.text_input("❓ Your Question")
    submitted = st.form_submit_button("Run RAG")

# ---------- ON SUBMIT ----------
if submitted:
    if not gemini_key or not urls_input or not user_question:
        st.error("❗ Please fill all fields above.")
    else:
        with st.spinner("📡 Loading content from web..."):
            urls = urls_input.strip().splitlines()
            docs = []
            for url in urls:
                try:
                    docs.extend(WebBaseLoader(url).load())
                except Exception as e:
                    st.warning(f"⚠️ Failed to load {url}: {e}")
            if not docs:
                st.error("❌ No documents loaded.")
                st.stop()

        with st.spinner("✂️ Chunking & embedding text..."):
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            chunks = splitter.split_documents(docs)
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            with tempfile.TemporaryDirectory() as tmpdir:
                vectorstore = FAISS.from_documents(chunks, embeddings)
                vectorstore.save_local(tmpdir)
                vectorstore = FAISS.load_local(tmpdir, embeddings)
                retriever = vectorstore.as_retriever()
                docs = retriever.get_relevant_documents(user_question)
                context = "\n\n".join(doc.page_content for doc in docs[:2])

        with st.spinner("🤖 Asking Gemini 2.0 Flash..."):
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={gemini_key}"
            prompt = f"Answer the question based on the context below:\n\nContext:\n{context}\n\nQuestion:\n{user_question}"
            payload = {"contents": [{"parts": [{"text": prompt}]}]}
            headers = {"Content-Type": "application/json"}
            res = requests.post(url, headers=headers, json=payload)

            try:
                answer = res.json()["candidates"][0]["content"]["parts"][0]["text"]
            except:
                answer = res.text or "❌ Gemini returned an error."

        # ---------- OUTPUT ----------
        st.markdown("### ✅ Gemini Answer")
        st.success(answer)

        with st.expander("📄 Retrieved Context"):
            st.code(context[:3000], language="text")