# Environment Variables Setup

## Overview
File .env digunakan untuk menyimpan API keys. File ini TIDAK di-commit ke Git.

## Setup

### 1. Copy Template
```bash
cp .env.example .env
```

### 2. Get Gemini API Key
Visit: https://makersuite.google.com/app/apikey

### 3. Edit .env
```env
GEMINI_API_KEY=your-actual-api-key-here
```

## Verify
```bash
python -c "from dotenv import load_dotenv; import os; load_dotenv(); print('Loaded:', bool(os.getenv('GEMINI_API_KEY')))"
```

## Security
- .env is in .gitignore (private)
- .env.example is template only (safe to share)
- Never commit actual keys to Git

## Files Updated
- examples/demo_gemini.py
- examples/test_gemini_interactive.py
- rag_with_llm.py

All now read from .env for security.
