import streamlit as st
import os
import sys
import tempfile
import re
import json
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime, timedelta

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent))

from src.vectorstore import FaissVectorStore
from src.database import (
    init_database, test_connection, 
    create_conversation, get_conversations, get_conversation,
    update_conversation_title, delete_conversation, archive_conversation,
    add_message, get_messages, search_conversations,
    add_bookmark, get_bookmarks, export_conversation,
    generate_title_from_message, add_tags_to_conversation
)
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from google import genai
from google.genai import types

st.set_page_config(
    page_title="RAG Chat System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Gemini client
def get_gemini_client():
    api_key = os.getenv("GEMINI_API_KEY")
    return genai.Client(api_key=api_key)

# Session state initialization
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'documents_loaded' not in st.session_state:
    st.session_state.documents_loaded = []
if 'document_contents' not in st.session_state:
    st.session_state.document_contents = {}
if 'current_conversation_id' not in st.session_state:
    st.session_state.current_conversation_id = None
if 'db_connected' not in st.session_state:
    st.session_state.db_connected = False
if 'conversations_list' not in st.session_state:
    st.session_state.conversations_list = []

def initialize_database():
    """Initialize database connection and tables"""
    success, message = test_connection()
    if success:
        init_success, init_message = init_database()
        st.session_state.db_connected = init_success
        return init_success, init_message
    else:
        st.session_state.db_connected = False
        return False, message

def load_conversation(conversation_id: int):
    """Load a conversation from database"""
    if not st.session_state.db_connected:
        return
    
    messages = get_messages(conversation_id)
    st.session_state.messages = [
        {
            "role": msg["role"],
            "content": msg["content"],
            "sources": msg.get("sources", []),
            "tool_calls": msg.get("tool_calls", []),
            "id": msg.get("id")
        }
        for msg in messages
    ]
    st.session_state.current_conversation_id = conversation_id

def save_message_to_db(role: str, content: str, sources: list = None, tool_calls: list = None):
    """Save a message to the database"""
    if not st.session_state.db_connected:
        return None
    
    # Create new conversation if none exists
    if st.session_state.current_conversation_id is None:
        title = generate_title_from_message(content) if role == "user" else "New Chat"
        conv_id = create_conversation(title=title, document_context=st.session_state.documents_loaded)
        if conv_id:
            st.session_state.current_conversation_id = conv_id
        else:
            return None
    
    # Add message to database
    msg_id = add_message(
        st.session_state.current_conversation_id,
        role, content, sources, tool_calls
    )
    
    # Update title if this is the first user message
    if role == "user" and len(st.session_state.messages) <= 1:
        title = generate_title_from_message(content)
        update_conversation_title(st.session_state.current_conversation_id, title)
    
    return msg_id

def start_new_conversation():
    """Start a new conversation"""
    st.session_state.messages = []
    st.session_state.current_conversation_id = None

def refresh_conversations_list():
    """Refresh the conversations list from database"""
    if st.session_state.db_connected:
        st.session_state.conversations_list = get_conversations(limit=50)

def format_timestamp(dt):
    """Format timestamp for display"""
    if not dt:
        return ""
    
    if isinstance(dt, str):
        try:
            dt = datetime.fromisoformat(dt.replace('Z', '+00:00'))
        except:
            return dt
    
    now = datetime.now(dt.tzinfo) if dt.tzinfo else datetime.now()
    diff = now - dt
    
    if diff < timedelta(minutes=1):
        return "Just now"
    elif diff < timedelta(hours=1):
        mins = int(diff.total_seconds() / 60)
        return f"{mins}m ago"
    elif diff < timedelta(days=1):
        hours = int(diff.total_seconds() / 3600)
        return f"{hours}h ago"
    elif diff < timedelta(days=7):
        days = diff.days
        return f"{days}d ago"
    else:
        return dt.strftime("%b %d, %Y")

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
        # Save to temp dir first for processing
        temp_path = os.path.join(temp_dir, uploaded_file.name)
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        file_extension = Path(uploaded_file.name).suffix.lower()
        
        # Also save to data/pdf or data/text_files permanently
        if file_extension == '.pdf':
            permanent_dir = Path("data/pdf")
            loader = PyPDFLoader(temp_path)
        elif file_extension == '.txt':
            permanent_dir = Path("data/text_files")
            loader = TextLoader(temp_path)
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")
        
        # Create permanent directory if not exists and save file
        permanent_dir.mkdir(parents=True, exist_ok=True)
        permanent_path = permanent_dir / uploaded_file.name
        
        # Copy to permanent location
        import shutil
        shutil.copy2(temp_path, permanent_path)
        
        docs = loader.load()
        
        if st.session_state.vector_store is None:
            st.session_state.vector_store = FaissVectorStore("faiss_store")
        
        st.session_state.vector_store.build_from_documents(docs)
        
        if uploaded_file.name not in st.session_state.documents_loaded:
            st.session_state.documents_loaded.append(uploaded_file.name)
        
        # Store full document content for full-document reading
        full_content = "\n\n".join([
            f"[Page {doc.metadata.get('page', i+1)}]\n{doc.page_content}"
            for i, doc in enumerate(docs)
        ])
        st.session_state.document_contents[uploaded_file.name] = {
            'content': full_content,
            'pages': len(docs),
            'metadata': docs[0].metadata if docs else {},
            'path': str(permanent_path)  # Store permanent path
        }
        
        return True, len(docs)
    except Exception as e:
        return False, str(e)

def format_chat_history(messages, max_turns=10):
    """Format recent chat history for context"""
    if not messages:
        return ""
    
    # Get last N turns (excluding the current question)
    recent = messages[-(max_turns * 2):] if len(messages) > max_turns * 2 else messages
    
    history_lines = []
    for msg in recent:
        role = "User" if msg["role"] == "user" else "Assistant"
        history_lines.append(f"{role}: {msg['content']}")
    
    return "\n".join(history_lines)

def is_conversational_query(question):
    """Check if the query is a greeting or conversational message"""
    greetings = [
        'hello', 'hi', 'hey', 'halo', 'hai', 'hei',
        'good morning', 'good afternoon', 'good evening', 'good night',
        'selamat pagi', 'selamat siang', 'selamat sore', 'selamat malam',
        'what\'s up', 'how are you', 'apa kabar', 'thanks', 'thank you',
        'terima kasih', 'bye', 'goodbye', 'sampai jumpa'
    ]
    question_lower = question.lower().strip()
    
    # Check if it's a short greeting
    if len(question_lower.split()) <= 3:
        for greeting in greetings:
            if greeting in question_lower:
                return True
    return False

# =============================================================================
# TOOL FUNCTIONS FOR AGENTIC RAG
# =============================================================================

def read_full_document(filename: str) -> dict:
    """
    Read the complete content of a specific document/paper.
    Use this when user asks to review, summarize, or analyze an entire paper.
    
    Args:
        filename: The name of the document file to read (e.g., '2211.03533v1.pdf')
    
    Returns:
        A dictionary containing the full document content and metadata.
    """
    # Try exact match first
    if filename in st.session_state.document_contents:
        doc = st.session_state.document_contents[filename]
        return {
            "status": "success",
            "filename": filename,
            "pages": doc['pages'],
            "content": doc['content']
        }
    
    # Try partial match
    for doc_name in st.session_state.document_contents:
        if filename.lower() in doc_name.lower() or doc_name.lower() in filename.lower():
            doc = st.session_state.document_contents[doc_name]
            return {
                "status": "success",
                "filename": doc_name,
                "pages": doc['pages'],
                "content": doc['content']
            }
    
    # If not in cache, try to rebuild from metadata
    if st.session_state.vector_store and st.session_state.vector_store.metadata:
        matching_chunks = []
        for meta in st.session_state.vector_store.metadata:
            source = os.path.basename(meta.get('source', ''))
            if filename.lower() in source.lower() or source.lower() in filename.lower():
                matching_chunks.append(meta)
        
        if matching_chunks:
            # Sort by page number
            matching_chunks.sort(key=lambda x: x.get('page', 0) or 0)
            content = "\n\n".join([
                f"[Page {c.get('page_label', c.get('page', 'N/A'))}]\n{c['text']}"
                for c in matching_chunks
            ])
            return {
                "status": "success",
                "filename": os.path.basename(matching_chunks[0]['source']),
                "pages": len(set(c.get('page', 0) for c in matching_chunks)),
                "content": content
            }
    
    return {
        "status": "error",
        "message": f"Document '{filename}' not found. Available documents: {list(st.session_state.document_contents.keys())}"
    }

def search_documents(query: str, num_results: int = 5) -> dict:
    """
    Search for relevant passages across all documents using semantic search.
    Use this for specific questions that need targeted information.
    
    Args:
        query: The search query to find relevant passages
        num_results: Number of results to return (default: 5)
    
    Returns:
        A dictionary containing search results with relevance scores.
    """
    if st.session_state.vector_store is None:
        return {"status": "error", "message": "No documents loaded"}
    
    results = st.session_state.vector_store.query(query, top_k=num_results)
    
    if not results:
        return {"status": "no_results", "message": "No relevant passages found"}
    
    formatted_results = []
    for r in results:
        formatted_results.append({
            "source": os.path.basename(r['source']),
            "page": r.get('page_label', r.get('page', 'N/A')),
            "relevance_score": r['similarity_score'],
            "content": r['text']
        })
    
    return {
        "status": "success",
        "query": query,
        "results": formatted_results
    }

def list_available_documents() -> dict:
    """
    List all available documents that have been loaded.
    Use this to see what documents are available for analysis.
    
    Returns:
        A dictionary containing list of available documents with their metadata.
    """
    if not st.session_state.documents_loaded:
        return {"status": "empty", "message": "No documents loaded yet"}
    
    docs_info = []
    for doc_name in st.session_state.documents_loaded:
        info = {"filename": doc_name}
        if doc_name in st.session_state.document_contents:
            info["pages"] = st.session_state.document_contents[doc_name]['pages']
        docs_info.append(info)
    
    return {
        "status": "success",
        "total_documents": len(docs_info),
        "documents": docs_info
    }

def get_document_section(filename: str, start_page: int, end_page: int) -> dict:
    """
    Get a specific section of a document by page range.
    Use this to read specific parts of a long document.
    
    Args:
        filename: The document filename
        start_page: Starting page number
        end_page: Ending page number
    
    Returns:
        A dictionary containing the requested section content.
    """
    if st.session_state.vector_store is None:
        return {"status": "error", "message": "No documents loaded"}
    
    matching_chunks = []
    for meta in st.session_state.vector_store.metadata:
        source = os.path.basename(meta.get('source', ''))
        if filename.lower() in source.lower():
            page = meta.get('page', 0) or 0
            if start_page <= page + 1 <= end_page:  # page is 0-indexed
                matching_chunks.append(meta)
    
    if not matching_chunks:
        return {"status": "error", "message": f"No content found for pages {start_page}-{end_page}"}
    
    matching_chunks.sort(key=lambda x: x.get('page', 0) or 0)
    content = "\n\n".join([
        f"[Page {c.get('page_label', c.get('page', 'N/A'))}]\n{c['text']}"
        for c in matching_chunks
    ])
    
    return {
        "status": "success",
        "filename": filename,
        "pages": f"{start_page}-{end_page}",
        "content": content
    }

def compare_documents(filename1: str, filename2: str, aspect: str) -> dict:
    """
    Compare two documents on a specific aspect.
    Use this when user wants to compare papers or documents.
    
    Args:
        filename1: First document filename
        filename2: Second document filename  
        aspect: The aspect to compare (e.g., 'methodology', 'results', 'approach')
    
    Returns:
        A dictionary containing content from both documents for comparison.
    """
    doc1 = read_full_document(filename1)
    doc2 = read_full_document(filename2)
    
    if doc1.get('status') == 'error':
        return doc1
    if doc2.get('status') == 'error':
        return doc2
    
    return {
        "status": "success",
        "aspect": aspect,
        "document1": {
            "filename": doc1['filename'],
            "content": doc1['content']
        },
        "document2": {
            "filename": doc2['filename'],
            "content": doc2['content']
        }
    }

# Tool function mapping
TOOL_FUNCTIONS = {
    "read_full_document": read_full_document,
    "search_documents": search_documents,
    "list_available_documents": list_available_documents,
    "get_document_section": get_document_section,
    "compare_documents": compare_documents
}

# =============================================================================
# AGENTIC RAG WITH FUNCTION CALLING
# =============================================================================

def get_tool_declarations():
    """Get function declarations for Gemini"""
    return [
        {
            "name": "read_full_document",
            "description": "Read the complete content of a specific document/paper. Use this when user asks to review, summarize, analyze, or critique an entire paper or document. Also use when user mentions a specific paper title or filename.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "The name of the document file to read (e.g., '2211.03533v1.pdf', 'sentiment_analysis.pdf')"
                    }
                },
                "required": ["filename"]
            }
        },
        {
            "name": "search_documents",
            "description": "Search for relevant passages across all documents using semantic search. Use this for specific factual questions that need targeted information from documents.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to find relevant passages"
                    },
                    "num_results": {
                        "type": "integer",
                        "description": "Number of results to return (default: 5)"
                    }
                },
                "required": ["query"]
            }
        },
        {
            "name": "list_available_documents",
            "description": "List all available documents that have been loaded. Use this when user asks what documents/papers are available.",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        },
        {
            "name": "get_document_section",
            "description": "Get a specific section of a document by page range. Use this to read specific parts like introduction, methodology, results, or conclusion.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "The document filename"
                    },
                    "start_page": {
                        "type": "integer",
                        "description": "Starting page number"
                    },
                    "end_page": {
                        "type": "integer",
                        "description": "Ending page number"
                    }
                },
                "required": ["filename", "start_page", "end_page"]
            }
        },
        {
            "name": "compare_documents",
            "description": "Compare two documents on a specific aspect. Use this when user wants to compare papers, methodologies, or approaches.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename1": {
                        "type": "string",
                        "description": "First document filename"
                    },
                    "filename2": {
                        "type": "string",
                        "description": "Second document filename"
                    },
                    "aspect": {
                        "type": "string",
                        "description": "The aspect to compare (e.g., 'methodology', 'results', 'approach', 'findings')"
                    }
                },
                "required": ["filename1", "filename2", "aspect"]
            }
        }
    ]

def execute_tool_call(function_call):
    """Execute a tool function and return the result"""
    func_name = function_call.name
    args = dict(function_call.args) if function_call.args else {}
    
    if func_name in TOOL_FUNCTIONS:
        try:
            result = TOOL_FUNCTIONS[func_name](**args)
            return result
        except Exception as e:
            return {"status": "error", "message": str(e)}
    else:
        return {"status": "error", "message": f"Unknown function: {func_name}"}

def agentic_rag_query(question: str, chat_history: str = "") -> tuple[str, list, list]:
    """
    Process a query using Agentic RAG with function calling.
    Returns: (answer, sources, tool_calls_made)
    """
    client = get_gemini_client()
    
    # Build the tools configuration
    tools = types.Tool(function_declarations=get_tool_declarations())
    
    # System instruction for the agent
    system_instruction = """You are an intelligent research assistant that helps users analyze academic papers and documents.

You have access to these tools:
1. read_full_document - Read entire papers (USE THIS for paper reviews, summaries, critiques)
2. search_documents - Search for specific information across documents
3. list_available_documents - See what documents are available
4. get_document_section - Read specific pages of a document
5. compare_documents - Compare two papers

IMPORTANT GUIDELINES:
- For "review paper X", "summarize X", "analyze X", "critique X" → Use read_full_document
- For specific factual questions → Use search_documents
- For "what papers do you have?" → Use list_available_documents
- For comparing papers → Use compare_documents

When reviewing a paper, provide:
1. Overview/Abstract summary
2. Key contributions
3. Methodology
4. Main findings/Results
5. Strengths and limitations
6. Conclusion

Always cite sources with [Source: filename, Page: X] format."""

    # Build conversation contents
    contents = []
    
    # Add chat history context if available
    if chat_history:
        contents.append(types.Content(
            role="user",
            parts=[types.Part(text=f"Previous conversation context:\n{chat_history}\n\nCurrent question: {question}")]
        ))
    else:
        contents.append(types.Content(
            role="user",
            parts=[types.Part(text=question)]
        ))
    
    config = types.GenerateContentConfig(
        tools=[tools],
        system_instruction=system_instruction,
        temperature=0.3
    )
    
    tool_calls_made = []
    sources = []
    max_iterations = 5
    iteration = 0
    
    while iteration < max_iterations:
        iteration += 1
        
        # Generate response
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=contents,
            config=config
        )
        
        # Check if there are function calls
        if response.candidates and response.candidates[0].content.parts:
            has_function_call = False
            
            for part in response.candidates[0].content.parts:
                if part.function_call:
                    has_function_call = True
                    function_call = part.function_call
                    
                    # Execute the function
                    tool_result = execute_tool_call(function_call)
                    tool_calls_made.append({
                        "function": function_call.name,
                        "args": dict(function_call.args) if function_call.args else {},
                        "result_status": tool_result.get("status", "unknown")
                    })
                    
                    # Track sources from search results
                    if function_call.name == "search_documents" and tool_result.get("results"):
                        for r in tool_result["results"]:
                            sources.append({
                                'source': r['source'],
                                'page': r['page'],
                                'score': r['relevance_score'],
                                'preview': r['content'][:200] + "..."
                            })
                    
                    # Add the model's response to contents
                    contents.append(response.candidates[0].content)
                    
                    # Add function response
                    function_response_part = types.Part.from_function_response(
                        name=function_call.name,
                        response={"result": tool_result}
                    )
                    contents.append(types.Content(
                        role="user",
                        parts=[function_response_part]
                    ))
                    break  # Process one function call at a time
            
            if not has_function_call:
                # No more function calls, return the text response
                return response.text, sources, tool_calls_made
        else:
            # No function calls in response
            if response.text:
                return response.text, sources, tool_calls_made
            else:
                return "I couldn't generate a response. Please try rephrasing your question.", sources, tool_calls_made
    
    return "Maximum iterations reached. Please try a simpler query.", sources, tool_calls_made

def generate_conversational_response(question, chat_history=""):
    """Generate a friendly response for greetings"""
    try:
        client = get_gemini_client()
        
        history_section = f"\nCONVERSATION HISTORY:\n{chat_history}\n" if chat_history else ""
        
        prompt = f"""You are a helpful research assistant for document analysis.
{history_section}
The user said: "{question}"

Respond naturally and briefly as a friendly assistant. 
If it's a greeting, greet back warmly and offer to help with document questions.
Keep your response SHORT (1-2 sentences max)."""
        
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        return response.text
    except Exception as e:
        return f"Error: {str(e)}"

def generate_answer(question, context, has_relevant_context=True, chat_history=""):
    """Legacy function for simple RAG queries (fallback)"""
    try:
        client = get_gemini_client()
        
        history_section = f"\nCONVERSATION HISTORY:\n{chat_history}\n" if chat_history else ""
        
        if not has_relevant_context:
            prompt = f"""You are a helpful research assistant for document analysis.
{history_section}
CURRENT QUESTION: "{question}"

No relevant documents were found for this query.
Respond helpfully: if it's a general question, answer briefly. 
If it needs document context, let them know you couldn't find relevant information and suggest rephrasing.
Use conversation history to understand context (e.g., "it", "that", "more details").
Keep response concise."""
        else:
            prompt = f"""You are a helpful research assistant. Answer based on the document context and conversation history.
{history_section}
DOCUMENT CONTEXT:
{context}

CURRENT QUESTION: {question}

Instructions:
- Use conversation history to understand references (e.g., "it", "that", "explain more")
- Provide clear answer with citations [Source: filename, Page: X]
- Be concise but comprehensive
"""
        
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
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

# Distance threshold - documents with distance above this are considered irrelevant
RELEVANCE_THRESHOLD = 1.0  # L2 distance threshold (lower = more strict)

def query_documents(question, top_k=3, use_agentic=True):
    """
    Main query function that routes to either Agentic RAG or simple RAG.
    """
    if st.session_state.vector_store is None:
        return None, [], []
    
    # Get conversation history for multi-turn context
    chat_history = format_chat_history(st.session_state.messages)
    
    # Check if it's a greeting/conversational query first
    if is_conversational_query(question):
        answer = generate_conversational_response(question, chat_history)
        return answer, [], []
    
    # Use Agentic RAG with function calling
    if use_agentic:
        try:
            answer, sources, tool_calls = agentic_rag_query(question, chat_history)
            return answer, sources, tool_calls
        except Exception as e:
            st.warning(f"Agentic RAG failed, falling back to simple RAG: {str(e)}")
            # Fall through to simple RAG
    
    # Fallback: Simple RAG
    try:
        results = st.session_state.vector_store.query(question, top_k=top_k)
        
        if not results:
            return generate_answer(question, "", has_relevant_context=False, chat_history=chat_history), [], []
        
        # Filter results by relevance threshold
        relevant_results = [
            r for r in results 
            if r['similarity_score'] < RELEVANCE_THRESHOLD
        ]
        
        # If no relevant results after filtering, respond without context
        if not relevant_results:
            all_sources = [{
                'source': os.path.basename(r['source']),
                'page': r.get('page_label', 'N/A'),
                'score': r['similarity_score'],
                'preview': r['text'][:200] + "..."
            } for r in results]
            answer = generate_answer(question, "", has_relevant_context=False, chat_history=chat_history)
            return answer, all_sources, []
        
        context = "\n\n".join([
            f"[{os.path.basename(r['source'])}, Page {r.get('page_label', 'N/A')}]\n{r['text']}"
            for r in relevant_results
        ])
        
        answer = generate_answer(question, context, has_relevant_context=True, chat_history=chat_history)
        
        sources = [{
            'source': os.path.basename(r['source']),
            'page': r.get('page_label', 'N/A'),
            'score': r['similarity_score'],
            'preview': r['text'][:200] + "..."
        } for r in results]
        
        return answer, sources, []
    except Exception as e:
        return f"Error: {str(e)}", [], []

def main():
    # Initialize database connection
    if not st.session_state.db_connected:
        db_success, db_message = initialize_database()
        if db_success:
            refresh_conversations_list()
    
    # Check API key
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or api_key == "your-new-gemini-api-key-here":
        st.error("⚠️ GEMINI_API_KEY not configured! Add it to your .env file.")
        st.stop()
    
    # Initialize vector store
    if st.session_state.vector_store is None:
        with st.spinner("Loading vector store..."):
            store, num_chunks = initialize_vector_store()
    
    # Rebuild document_contents from existing vector store metadata
    if st.session_state.vector_store and st.session_state.vector_store.metadata:
        if not st.session_state.document_contents:
            docs_chunks = {}
            for meta in st.session_state.vector_store.metadata:
                source = os.path.basename(meta.get('source', 'unknown'))
                if source not in docs_chunks:
                    docs_chunks[source] = []
                docs_chunks[source].append(meta)
            
            for source, chunks in docs_chunks.items():
                chunks.sort(key=lambda x: x.get('page', 0) or 0)
                content = "\n\n".join([
                    f"[Page {c.get('page_label', c.get('page', 'N/A'))}]\n{c['text']}"
                    for c in chunks
                ])
                st.session_state.document_contents[source] = {
                    'content': content,
                    'pages': len(set(c.get('page', 0) for c in chunks)),
                    'metadata': {}
                }
    
    # =========================================================================
    # SIDEBAR - Chat History & Settings
    # =========================================================================
    with st.sidebar:
        # New Chat Button
        if st.button("➕ New Chat", use_container_width=True, type="primary"):
            start_new_conversation()
            st.rerun()
        
        st.markdown("---")
        
        # Chat History Section
        st.header("💬 Chat History")
        
        if st.session_state.db_connected:
            # Search conversations
            search_query = st.text_input("🔍 Search chats", placeholder="Search...", key="chat_search")
            
            if search_query:
                conversations = search_conversations(search_query)
            else:
                conversations = st.session_state.conversations_list
            
            if conversations:
                # Group by date
                today = datetime.now().date()
                yesterday = today - timedelta(days=1)
                last_week = today - timedelta(days=7)
                
                today_convs = []
                yesterday_convs = []
                week_convs = []
                older_convs = []
                
                for conv in conversations:
                    conv_date = conv['updated_at']
                    if isinstance(conv_date, str):
                        try:
                            conv_date = datetime.fromisoformat(conv_date.replace('Z', '+00:00'))
                        except:
                            older_convs.append(conv)
                            continue
                    
                    conv_day = conv_date.date() if hasattr(conv_date, 'date') else today
                    
                    if conv_day == today:
                        today_convs.append(conv)
                    elif conv_day == yesterday:
                        yesterday_convs.append(conv)
                    elif conv_day > last_week:
                        week_convs.append(conv)
                    else:
                        older_convs.append(conv)
                
                # Display grouped conversations
                def display_conv_group(convs, label):
                    if convs:
                        st.caption(label)
                        for conv in convs:
                            col1, col2 = st.columns([5, 1])
                            with col1:
                                is_current = conv['id'] == st.session_state.current_conversation_id
                                btn_type = "primary" if is_current else "secondary"
                                
                                title = conv['title'][:30] + "..." if len(conv['title']) > 30 else conv['title']
                                
                                if st.button(
                                    f"{'📌 ' if is_current else ''}{title}",
                                    key=f"conv_{conv['id']}",
                                    use_container_width=True,
                                    type=btn_type
                                ):
                                    load_conversation(conv['id'])
                                    st.rerun()
                            
                            with col2:
                                if st.button("🗑️", key=f"del_{conv['id']}", help="Delete"):
                                    delete_conversation(conv['id'])
                                    if conv['id'] == st.session_state.current_conversation_id:
                                        start_new_conversation()
                                    refresh_conversations_list()
                                    st.rerun()
                
                display_conv_group(today_convs, "📅 Today")
                display_conv_group(yesterday_convs, "📅 Yesterday")
                display_conv_group(week_convs, "📅 This Week")
                display_conv_group(older_convs, "📅 Older")
            else:
                st.info("No conversations yet. Start a new chat!")
        else:
            st.warning("⚠️ Database not connected. Chat history disabled.")
            st.caption("Set DATABASE_URL in .env file to enable.")
        
        st.markdown("---")
        
        # Document Management Section
        with st.expander("📁 Document Management", expanded=False):
            uploaded_files = st.file_uploader(
                "Upload PDF or TXT",
                type=['pdf', 'txt'],
                accept_multiple_files=True,
                help="Limit 200MB per file",
                key="file_uploader"
            )
            
            if uploaded_files:
                if st.button("Process Uploads", type="primary", key="process_btn"):
                    with tempfile.TemporaryDirectory() as temp_dir:
                        progress = st.progress(0)
                        for idx, file in enumerate(uploaded_files):
                            success, result = process_uploaded_file(file, temp_dir)
                            if success:
                                st.success(f"✅ {file.name}: {result} pages")
                            else:
                                st.error(f"❌ {file.name}: {result}")
                            progress.progress((idx + 1) / len(uploaded_files))
                        st.rerun()
            
            if st.session_state.documents_loaded:
                st.caption("**Loaded Documents:**")
                for doc in st.session_state.documents_loaded:
                    pages = st.session_state.document_contents.get(doc, {}).get('pages', '?')
                    st.write(f"• {doc} ({pages} pages)")
            
            if st.session_state.vector_store and st.session_state.vector_store.metadata:
                st.metric("Total Chunks", len(st.session_state.vector_store.metadata))
        
        st.markdown("---")
        
        # Settings Section
        with st.expander("⚙️ Settings", expanded=False):
            use_agentic = st.toggle(
                "🤖 Agentic RAG", 
                value=True,
                help="Enable function calling for paper reviews & full document reading",
                key="agentic_toggle"
            )
            
            top_k = st.slider("Sources to retrieve", 1, 10, 5, key="top_k_slider")
            
            st.markdown("---")
            
            # Database status
            if st.session_state.db_connected:
                st.success("✅ Database connected")
            else:
                st.error("❌ Database disconnected")
                if st.button("🔄 Retry Connection"):
                    db_success, db_message = initialize_database()
                    if db_success:
                        refresh_conversations_list()
                        st.success("Connected!")
                    else:
                        st.error(f"Failed: {db_message}")
            
            # Export current conversation
            if st.session_state.current_conversation_id and st.session_state.db_connected:
                st.markdown("---")
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("📥 Export MD"):
                        md_content = export_conversation(st.session_state.current_conversation_id, "markdown")
                        if md_content:
                            st.download_button(
                                "Download",
                                md_content,
                                file_name="conversation.md",
                                mime="text/markdown"
                            )
                with col2:
                    if st.button("📥 Export JSON"):
                        json_content = export_conversation(st.session_state.current_conversation_id, "json")
                        if json_content:
                            st.download_button(
                                "Download",
                                json_content,
                                file_name="conversation.json",
                                mime="application/json"
                            )
    
    # =========================================================================
    # MAIN CHAT AREA
    # =========================================================================
    
    # Header
    st.title("🤖 RAG Chat System")
    
    # Show current conversation info
    if st.session_state.current_conversation_id and st.session_state.db_connected:
        conv = get_conversation(st.session_state.current_conversation_id)
        if conv:
            st.caption(f"📝 {conv['title']} • {format_timestamp(conv['updated_at'])}")
    else:
        st.caption("💡 Start a new conversation by typing below")
    
    st.markdown("---")
    
    # Display chat messages
    for idx, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            if message["role"] == "assistant":
                # Tool calls expander
                if message.get("tool_calls"):
                    with st.expander("🔧 Tools Used"):
                        for tc in message["tool_calls"]:
                            st.write(f"**{tc['function']}**")
                            if tc.get('args'):
                                st.json(tc['args'])
                            st.caption(f"Status: {tc.get('result_status', 'unknown')}")
                            st.markdown("---")
                
                # Sources expander
                if message.get("sources"):
                    with st.expander("📄 View Sources"):
                        for src_idx, src in enumerate(message["sources"], 1):
                            st.write(f"**{src_idx}. {src['source']}** - Page {src['page']}")
                            if 'score' in src:
                                st.write(f"Distance: {src['score']:.3f} - {get_relevance_indicator(src['score'])}")
                            st.write(src.get('preview', ''))
                            st.markdown("---")
                
                # Bookmark button
                if st.session_state.db_connected and message.get('id'):
                    if st.button("⭐ Bookmark", key=f"bookmark_{idx}"):
                        add_bookmark(message['id'])
                        st.toast("Bookmarked!")
    
    # Chat input
    if prompt := st.chat_input("Ask about your documents... (e.g., 'Review paper 2211.03533v1.pdf')"):
        if not st.session_state.vector_store or not st.session_state.vector_store.metadata:
            st.warning("⚠️ Please upload documents first!")
            return
        
        # Add user message to session
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Save to database
        user_msg_id = save_message_to_db("user", prompt)
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            # Get settings from sidebar
            use_agentic = st.session_state.get('agentic_toggle', True)
            top_k = st.session_state.get('top_k_slider', 5)
            
            with st.spinner("🔍 Analyzing..." if use_agentic else "💭 Thinking..."):
                answer, sources, tool_calls = query_documents(prompt, top_k=top_k, use_agentic=use_agentic)
                st.markdown(answer)
                
                # Show tool calls
                if tool_calls:
                    with st.expander("🔧 Tools Used"):
                        for tc in tool_calls:
                            st.write(f"**{tc['function']}**")
                            if tc.get('args'):
                                st.json(tc['args'])
                            st.caption(f"Status: {tc.get('result_status', 'unknown')}")
                            st.markdown("---")
                
                # Show sources
                if sources:
                    with st.expander("📄 View Sources"):
                        for src_idx, src in enumerate(sources, 1):
                            st.write(f"**{src_idx}. {src['source']}** - Page {src['page']}")
                            if 'score' in src:
                                st.write(f"Distance: {src['score']:.3f} - {get_relevance_indicator(src['score'])}")
                            st.write(src.get('preview', ''))
                            st.markdown("---")
        
        # Save assistant response to session and database
        assistant_msg = {
            "role": "assistant",
            "content": answer,
            "sources": sources,
            "tool_calls": tool_calls
        }
        st.session_state.messages.append(assistant_msg)
        
        assistant_msg_id = save_message_to_db("assistant", answer, sources, tool_calls)
        if assistant_msg_id:
            st.session_state.messages[-1]['id'] = assistant_msg_id
        
        # Refresh conversations list
        refresh_conversations_list()
    
    # Welcome message for new conversations
    if not st.session_state.messages:
        st.markdown("""
### 👋 Welcome to Agentic RAG Chat System!

**Features:**
- 🤖 **Agentic RAG**: Automatically reads full documents for paper reviews
- 🔍 **Smart Search**: Semantic search for specific questions  
- 📚 **Multi-turn**: Remembers conversation context
- 💾 **Chat History**: All conversations saved to database
- ⭐ **Bookmarks**: Save important answers for later

**Quick Start:**
1. Upload documents (PDF/TXT) in the sidebar
2. Ask questions like:
   - *"Review paper 2211.03533v1.pdf"*
   - *"Summarize the methodology section"*
   - *"What are the main findings?"*
   - *"Compare paper A with paper B"*

**Tips:**
- Use **New Chat** to start a fresh conversation
- Click on previous chats in the sidebar to continue them
- Toggle **Agentic RAG** for full document reading capability
        """)

if __name__ == "__main__":
    main()
