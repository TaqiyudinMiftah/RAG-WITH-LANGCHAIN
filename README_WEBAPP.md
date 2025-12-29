# ��� RAG Web Application - Deployment Guide

Web-based RAG system dengan Streamlit untuk upload documents dan chat dengan citations.

## ✨ Features

- **��� File Upload**: Upload PDF atau TXT files
- **��� Smart Search**: Vector-based semantic search
- **��� Interactive Chat**: Conversational interface dengan chat history
- **��� Source Citations**: Setiap jawaban disertai sources dan page numbers
- **��� Document Management**: Track uploaded documents dan statistics
- **⚙️ Customizable**: Adjust number of sources retrieved

---

## ���️ Setup

### 1. Install Dependencies

```bash
# Make sure all dependencies installed
uv pip install -r requirements.txt
```

### 2. Configure API Key

Pastikan file `.env` sudah ada dengan Gemini API key:

```bash
GEMINI_API_KEY=your-actual-api-key-here
```

Get your API key from: https://makersuite.google.com/app/apikey

### 3. (Optional) Pre-load Existing Vector Store

Jika sudah punya vector store:

```bash
python scripts/rebuild_with_metadata.py
```

Atau biarkan kosong, upload via web interface nanti.

---

## ��� Run Application

### Method 1: Direct Streamlit Command

```bash
streamlit run app.py
```

### Method 2: Using Python

```bash
python -m streamlit run app.py
```

### Method 3: With Custom Port

```bash
streamlit run app.py --server.port 8080
```

Application will open automatically di browser: **http://localhost:8501**

---

## ��� How to Use

### 1. Upload Documents

1. Klik sidebar "Document Management"
2. Click "Browse files" atau drag & drop PDF/TXT files
3. Click "��� Process Uploads"
4. Wait for processing (akan tampil progress bar)
5. Documents akan di-chunk dan di-embed ke vector store

### 2. Ask Questions

1. Type pertanyaan di chat input
2. Click Enter atau Send
3. System akan:
   - Retrieve relevant chunks
   - Generate answer menggunakan Gemini
   - Display answer dengan citations
4. Click "��� View Sources" untuk lihat source documents

### 3. Manage Settings

- **Sidebar → Statistics**: Lihat total chunks dan documents
- **Sidebar → Settings**: Adjust number of sources (top_k)
- **Sidebar → Clear Chat**: Reset conversation history

---

## ��� Interface Overview

```
┌─────────────────────────────────────────────────────────┐
│                    ��� RAG Chat System                    │
├───────────────┬─────────────────────────────────────────┤
│   SIDEBAR     │          MAIN CHAT AREA                  │
│               │                                          │
│ Document Mgmt │  ��� Chat with your documents...         │
│ • Upload      │                                          │
│ • List        │  User: What is RAG?                     │
│ • Stats       │                                          │
│               │  Assistant: RAG is...                    │
│ Settings      │  ��� View Sources                         │
│ • top_k       │    Source 1: paper.pdf - Page 3         │
│ • Clear chat  │    Relevance: 95%                       │
│               │                                          │
└───────────────┴─────────────────────────────────────────┘
```

---

## ��� Example Usage

### Example 1: Research Paper Analysis

1. Upload: `research_paper.pdf`
2. Ask: "What is the main contribution of this paper?"
3. Get answer dengan citations: [Source: research_paper.pdf, Page: 3]

### Example 2: Multi-Document QA

1. Upload multiple PDFs
2. Ask: "Compare the methods used in different papers"
3. Get consolidated answer dari multiple sources

### Example 3: Fact Extraction

1. Upload documents
2. Ask: "List all evaluation metrics mentioned"
3. Get bullet-point answer dengan page references

---

## ��� Troubleshooting

### Issue: "GEMINI_API_KEY not configured"

**Solution**: 
```bash
# Edit .env file
GEMINI_API_KEY=your-api-key-here
```

### Issue: "No relevant documents found"

**Solution**: Upload documents first via sidebar

### Issue: Port already in use

**Solution**: 
```bash
# Use different port
streamlit run app.py --server.port 8080
```

### Issue: Out of memory

**Solution**: 
- Reduce chunk size di `src/vectorstore.py`
- Process fewer documents at once
- Consider using IndexIVFFlat untuk large datasets

---

## ��� Deployment Options

### Option 1: Streamlit Cloud (Recommended)

1. Push code ke GitHub
2. Go to: https://streamlit.io/cloud
3. Connect repository
4. Add secrets (GEMINI_API_KEY)
5. Deploy!

**Pros**: Free, easy, auto-scaling
**Cons**: Public URL, limited resources

### Option 2: Docker

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY . .

RUN pip install -r requirements.txt

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

```bash
docker build -t rag-app .
docker run -p 8501:8501 --env-file .env rag-app
```

### Option 3: Cloud VM (GCP/AWS/Azure)

```bash
# SSH to VM
git clone your-repo
cd RAG-With-Langchain

# Setup
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run with nohup
nohup streamlit run app.py --server.port 80 &
```

### Option 4: Railway/Render/Fly.io

Similar to Streamlit Cloud, with more control dan resources.

---

## ��� Performance Tips

### 1. Pre-build Vector Store

```bash
# Build once, load many times
python scripts/rebuild_with_metadata.py
```

### 2. Use Caching

Streamlit automatically caches `@st.cache_data` functions.

### 3. Optimize Parameters

- Reduce `top_k` for faster responses
- Use smaller embedding models
- Implement pagination for large document lists

### 4. Scale Vector Store

When > 100K chunks:
```python
# In src/vectorstore.py
# Switch to IndexHNSWFlat
index = faiss.IndexHNSWFlat(d, 32)
```

---

## ��� Security Considerations

### For Production:

1. **API Key Management**: Use environment variables, never commit keys
2. **File Upload Validation**: Already implemented (PDF/TXT only)
3. **Rate Limiting**: Add Streamlit rate limiting for public deployment
4. **Authentication**: Add login system for sensitive documents
5. **HTTPS**: Always use HTTPS in production

### Example: Add Authentication

```python
# In app.py
def check_password():
    def password_entered():
        if st.session_state["password"] == os.getenv("APP_PASSWORD"):
            st.session_state["password_correct"] = True
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.text_input("Password", type="password", on_change=password_entered, key="password")
        return False
    elif not st.session_state["password_correct"]:
        st.text_input("Password", type="password", on_change=password_entered, key="password")
        st.error("��� Password incorrect")
        return False
    else:
        return True

if not check_password():
    st.stop()
```

---

## ��� Customization

### Change LLM Provider

Edit `generate_answer()` in [app.py](app.py):

```python
# For OpenAI
from openai import OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
response = client.chat.completions.create(...)

# For Ollama (local)
import ollama
response = ollama.generate(model="llama2", prompt=prompt)
```

### Add More File Types

Edit `process_uploaded_file()`:

```python
elif file_extension == '.docx':
    from langchain_community.document_loaders import Docx2txtLoader
    loader = Docx2txtLoader(file_path)
elif file_extension == '.csv':
    from langchain_community.document_loaders import CSVLoader
    loader = CSVLoader(file_path)
```

### Customize UI Theme

```python
# In app.py, modify st.markdown() CSS
st.markdown("""
<style>
    :root {
        --primary-color: #FF6B6B;  /* Red theme */
    }
    .main-header {
        color: #FF6B6B;
    }
</style>
""", unsafe_allow_html=True)
```

---

## ��� Next Steps

- [ ] Add user authentication
- [ ] Implement document deletion
- [ ] Add export chat history
- [ ] Multi-language support
- [ ] Advanced filtering (by date, author, etc)
- [ ] Visualization of retrieved chunks
- [ ] A/B testing different prompts
- [ ] Analytics dashboard

---

## ��� Support

Issues atau questions? Open issue di GitHub repository.

**Happy Deploying! ���**
