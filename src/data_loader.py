from pathlib import Path
from typing import List, Any
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders.excel import UnstructuredExcelLoader
from langchain_community.document_loaders import JSONLoader

def load_all_documents(data_dir: str) -> List[Any]:
    """Load documents from various file types in the specified directory."""
    loaders = {
        '.pdf': PyPDFLoader,
        '.txt': TextLoader,
        '.csv': CSVLoader,
        '.docx': Docx2txtLoader,
        '.xlsx': UnstructuredExcelLoader,
        '.json': JSONLoader,
    }

    documents = []
    data_path = Path(data_dir)

    for file_path in data_path.rglob('*'):
        if file_path.suffix in loaders:
            loader_class = loaders[file_path.suffix]
            loader = loader_class(str(file_path))
            docs = loader.load()
            documents.extend(docs)

    return documents