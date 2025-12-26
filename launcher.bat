@echo off
REM Quick Start Launcher untuk RAG System (Windows)
REM Script untuk memudahkan akses ke berbagai fungsi RAG

echo ========================================================================
echo 🚀 RAG WITH LANGCHAIN - QUICK LAUNCHER
echo ========================================================================
echo.
echo Pilih opsi yang ingin Anda jalankan:
echo.
echo 1. 💬 Interactive Chat         - Tanya jawab dengan paper collection
echo 2. 📚 Build Vector Store        - Index paper PDF ke database
echo 3. 🔍 Inspect Vector Store      - Lihat isi vector store
echo 4. 📊 Paper Review Demo         - Demo review paper
echo 5. 🧪 Test RAG with LLM         - Test RAG dengan berbagai LLM
echo 6. ❌ Exit
echo.
echo ========================================================================
echo.

set /p choice="Masukkan pilihan (1-6): "

if "%choice%"=="1" (
    echo.
    echo 🚀 Launching Interactive Chat...
    echo ========================================================================
    uv run python chat_with_rag.py
    goto end
)

if "%choice%"=="2" (
    echo.
    echo 🚀 Building Vector Store...
    echo ========================================================================
    echo ℹ️  Pastikan PDF paper ada di folder: data/pdf/
    echo.
    uv run python scripts/rebuild_with_metadata.py
    goto end
)

if "%choice%"=="3" (
    echo.
    echo 🚀 Inspecting Vector Store...
    echo ========================================================================
    uv run python scripts/inspect_metadata.py
    goto end
)

if "%choice%"=="4" (
    echo.
    echo 🚀 Running Paper Review Demo...
    echo ========================================================================
    uv run python examples/paper_review_rag.py
    goto end
)

if "%choice%"=="5" (
    echo.
    echo 🚀 Testing RAG with LLM...
    echo ========================================================================
    uv run python examples/rag_with_llm.py
    goto end
)

if "%choice%"=="6" (
    echo.
    echo 👋 Goodbye!
    goto end
)

echo.
echo ❌ Invalid option: %choice%
echo Please choose 1-6

:end
pause
