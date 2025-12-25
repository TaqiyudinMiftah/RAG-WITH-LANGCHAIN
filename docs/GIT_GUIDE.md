# Git & GitHub Guide

## Pre-Push Checklist

Sebelum push ke GitHub, pastikan file-file berikut **SUDAH DI-IGNORE**:

### í´´ CRITICAL (Security)
- [x] `.env` - API keys dan secrets
- [x] `*.key`, `*.pem` - Private keys
- [x] `config/secrets.yaml` - Secret configs

### í¿¡ IMPORTANT (Large Files)
- [x] `faiss_store/` - Vector store (bisa di-rebuild)
- [x] `faiss_store_backup/` - Backup vector store
- [x] `data/vector_store/` - ChromaDB
- [x] `*.pkl`, `*.index`, `*.faiss` - Index files
- [x] `archive/` - Backup files

### í¿¢ OPTIONAL (Data Files)
- [ ] `data/pdf/*.pdf` - PDF papers (jika private/large)
- [ ] `data/text_files/*.txt` - Text files (jika private/large)

### í´µ AUTOMATIC (Python/IDE)
- [x] `__pycache__/` - Python cache
- [x] `.venv/` - Virtual environment
- [x] `.vscode/`, `.idea/` - IDE configs
- [x] `*.pyc`, `*.pyo` - Compiled Python
- [x] `.ipynb_checkpoints/` - Jupyter checkpoints

## Verify Ignored Files

```bash
# Check if .env is ignored
git check-ignore -v .env

# List all ignored files
git status --ignored

# Show what will be committed
git status
```

## Safe to Commit

File-file ini **AMAN** untuk di-commit:

### âœ… Source Code
- `src/*.py` - Source code
- `examples/*.py` - Example scripts
- `scripts/*.py` - Utility scripts
- `paper_review_rag.py`, `rag_with_llm.py` - Main scripts

### âœ… Documentation
- `README.md` - Project overview
- `docs/*.md` - All documentation
- `.env.example` - Template (NO actual keys!)

### âœ… Configuration
- `requirements.txt` - Dependencies
- `pyproject.toml` - Project config
- `.gitignore` - Git ignore rules
- `uv.lock` - UV lockfile

### âœ… Data Structure (Empty or Sample)
- `data/pdf/` - Folder structure (but maybe not PDFs)
- `data/text_files/` - Folder structure (but maybe not large files)

## First Time Setup

```bash
# 1. Verify .gitignore is working
git status

# 2. Add safe files
git add .gitignore
git add .env.example
git add requirements.txt pyproject.toml
git add README.md docs/
git add src/ examples/ scripts/
git add paper_review_rag.py rag_with_llm.py

# 3. Commit
git commit -m "Initial commit: RAG system with Langchain"

# 4. Push to GitHub
git push origin master
```

## Update Existing Repo

```bash
# 1. Stage changes
git add .

# 2. Check what will be committed (VERIFY NO .env!)
git status

# 3. Commit
git commit -m "Add documentation and examples"

# 4. Push
git push
```

## If You Accidentally Committed .env

```bash
# Remove from Git but keep local file
git rm --cached .env

# Commit the removal
git commit -m "Remove .env from tracking"

# Push
git push
```

## Recommended .gitignore Sections

Our `.gitignore` covers:
1. **Security** - API keys, secrets
2. **Python** - Cache, compiled files
3. **Virtual Envs** - .venv, venv, etc
4. **Vector Stores** - Large index files
5. **Notebooks** - Jupyter checkpoints
6. **IDEs** - VS Code, PyCharm, etc
7. **OS** - macOS, Windows, Linux temp files
8. **Logs** - *.log, *.tmp
9. **Backups** - *.bak, archive/

## Quick Reference

```bash
# What's staged?
git diff --staged

# What's not tracked?
git ls-files --others --exclude-standard

# What's ignored?
git status --ignored

# What will be pushed?
git log origin/master..HEAD
```

## Repository Size Tips

Jika repo terlalu besar (>100MB), consider:
1. Ignore PDF files: Uncomment in `.gitignore`
2. Use Git LFS for large files
3. Keep vector stores local only (already ignored)
4. Archive old data files

## GitHub Secrets

Untuk CI/CD, tambahkan secrets di GitHub:
1. Go to repo Settings â†’ Secrets â†’ Actions
2. Add `GEMINI_API_KEY`
3. Use in workflows: `${{ secrets.GEMINI_API_KEY }}`
