# RAG Web Application - Quick Start Guide

## Features
- Upload PDF/TXT documents
- Chat with your documents
- Get answers with source citations
- Track document statistics

## Quick Start

### 1. Run the Application

**Windows:**
```bash
run_webapp.bat
```

**Linux/Mac:**
```bash
bash run_webapp.sh
```

**Or directly:**
```bash
uv run streamlit run app.py
```

The app will open at: http://localhost:8501

### 2. Use the Application

1. **Upload Documents**
   - Click sidebar "Upload PDF or TXT"
   - Select files
   - Click "Process Uploads"
   
2. **Ask Questions**
   - Type in chat input
   - Get AI-powered answers
   - View source citations

3. **Manage Settings**
   - Adjust number of sources
   - Clear chat history
   - View statistics

## Deployment Options

### Streamlit Cloud (Free)
1. Push to GitHub
2. Go to https://streamlit.io/cloud
3. Connect repository
4. Add GEMINI_API_KEY secret
5. Deploy!

### Docker
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
EXPOSE 8501
CMD ["streamlit", "run", "app.py"]
```

### Cloud VM
```bash
git clone your-repo
pip install -r requirements.txt
streamlit run app.py --server.port 80
```

## Troubleshooting

**API Key Error:**
- Set GEMINI_API_KEY in .env file

**No Documents:**
- Upload files via sidebar first

**Port in Use:**
- Use: `streamlit run app.py --server.port 8080`

Happy chatting!
