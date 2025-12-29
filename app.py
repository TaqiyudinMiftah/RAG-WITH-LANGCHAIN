import streamlit as st
import os
import sys
import tempfile
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent))

from src.vectorstore import FaissVectorStore
from langchain_community.document_loaders import PyPDFLoader, TextLoader
import google.generativeai as genai

st.set_page_config(
    page_title="RAG Chat System",
    page_icon="robot",
    layout="wide",
    initial_sidebar_state="expanded"
)

if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'documents_loaded' not in st.session_state:
    st.session_state.documents_loaded = []

def initialize_vector_store():
    try:
        store = FaissVectorStore("faiss_store")
        if os.path.exists("faiss_store/faiss_index.bin"):
            store.load()
            st.session_state.vector_store = store
            if store.metadata:
                sources = set([os.path.basename(m['source']) for m in store.metadata])
                st.session_state.documents_loaded = list(sources)
            return store, len(store.metadata)
        else:
            st.session_state.vector_store = store
            return store, 0
    except Exception as e:
        st.error(f"Error loading vector store: {str(e)}")
        return None, 0

def process_uploaded_file(uploaded_file, temp_dir):
    try:
        file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        file_extension = Path(uploaded_file.name).suffix.lower()
        
        if file_extension == '.pdf':
            loader = PyPDFLoader(file_path)
        elif file_extension == '.txt':
            loader = TextLoader(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")
        
        docs = loader.load()
        
        if st.session_state.vector_store is None:
            st.session_state.vector_store = FaissVectorStore("faiss_store")
        
        st.session_state.vector_store.build_from_documents(docs)
        
        if uploaded_file.name not in st.session_state.documents_loaded:
            st.session_state.documents_loaded.append(uploaded_file.name)
        
        return True, len(docs)
    except Exception as e:
        return False, str(e)

def generate_answer(question, context):
    try:
        api_key = os.getenv("GEMINI_API_KEY")
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        prompt = f"""Answer based on context:

CONTEXT:
{context}

QUESTION: {question}

Provide clear answer with citations [Source: filename, Page: X]
"""
        
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error: {str(e)}"

def get_relevance_indicator(distance):
    """Get visual indicator for L2 distance"""
    if distance < 0.3:
        return "🟢 Highly Relevant"
    elif distance < 0.6:
        return "🟡 Relevant"
    elif distance < 1.0:
        return "🟠 Moderately Relevant"
    else:
        return "🔴 Less Relevant"

def query_documents(question, top_k=3):
    if st.session_state.vector_store is None:
        return None, []
    
    try:
        results = st.session_state.vector_store.query(question, top_k=top_k)
        
        if not results:
            return "No relevant documents found.", []
        
        context = "\n\n".join([
            f"[{os.path.basename(r['source'])}, Page {r.get('page_label', 'N/A')}]\n{r['text']}"
            for r in results
        ])
        
        answer = generate_answer(question, context)
        
        sources = [{
            'source': os.path.basename(r['source']),
            'page': r.get('page_label', 'N/A'),
            'score': r['similarity_score'],
            'preview': r['text'][:200] + "..."
        } for r in results]
        
        return answer, sources
    except Exception as e:
        return f"Error: {str(e)}", []

def main():
    st.title("RAG Chat System")
    st.markdown("---")
    
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or api_key == "your-new-gemini-api-key-here":
        st.error("GEMINI_API_KEY not configured!")
        st.stop()
    
    if st.session_state.vector_store is None:
        with st.spinner("Loading..."):
            store, num_chunks = initialize_vector_store()
            if store and num_chunks > 0:
                st.success(f"Loaded {num_chunks} chunks")
    
    with st.sidebar:
        st.header("Document Management")
        
        uploaded_files = st.file_uploader(
            "Upload PDF or TXT",
            type=['pdf', 'txt'],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            if st.button("Process Uploads", type="primary"):
                with tempfile.TemporaryDirectory() as temp_dir:
                    progress = st.progress(0)
                    for idx, file in enumerate(uploaded_files):
                        success, result = process_uploaded_file(file, temp_dir)
                        if success:
                            st.success(f"{file.name}: {result} pages")
                        else:
                            st.error(f"{file.name}: {result}")
                        progress.progress((idx + 1) / len(uploaded_files))
                    st.rerun()
        
        st.subheader("Loaded Documents")
        if st.session_state.documents_loaded:
            for doc in st.session_state.documents_loaded:
                st.write(f"- {doc}")
        else:
            st.info("No documents yet")
        
        if st.session_state.vector_store and st.session_state.vector_store.metadata:
            st.subheader("Statistics")
            st.metric("Chunks", len(st.session_state.vector_store.metadata))
        
        top_k = st.slider("Sources to retrieve", 1, 10, 3)
        
        if st.button("Clear Chat"):
            st.session_state.messages = []
            st.rerun()
    
    st.header("Chat")
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and "sources" in message:
                with st.expander("View Sources"):
                    for idx, src in enumerate(message["sources"], 1):
                        st.write(f"**{idx}. {src['source']}** - Page {src['page']}")
                        st.write(f"Distance: {src['score']:.3f} - {get_relevance_indicator(src['score'])}")
                        st.write(src['preview'])
                        st.markdown("---")
    
    if prompt := st.chat_input("Ask about your documents..."):
        if not st.session_state.vector_store or not st.session_state.vector_store.metadata:
            st.warning("Upload documents first!")
            return
        
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer, sources = query_documents(prompt, top_k=top_k)
                st.markdown(answer)
                
                if sources:
                    with st.expander("View Sources"):
                        for idx, src in enumerate(sources, 1):
                            st.write(f"**{idx}. {src['source']}** - Page {src['page']}")
                            st.write(f"Distance: {src['score']:.3f} - {get_relevance_indicator(src['score'])}")
                            st.write(src['preview'])
                            st.markdown("---")
        
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources
        })
    
    if not st.session_state.messages:
        st.info("""
Welcome! 

1. Upload documents (PDF/TXT) in sidebar
2. Click Process Uploads
3. Ask questions
4. Get answers with citations
        """)

if __name__ == "__main__":
    main()
