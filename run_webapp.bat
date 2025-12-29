@echo off
echo ========================================
echo Starting RAG Web Application
echo ========================================
echo.
echo Opening browser at: http://localhost:8501
echo Press Ctrl+C to stop the server
echo.

uv run streamlit run app.py
