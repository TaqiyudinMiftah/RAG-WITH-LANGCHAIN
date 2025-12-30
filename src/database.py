# -*- coding: utf-8 -*-
"""
PostgreSQL Database Module for Chat History
Provides persistent storage for conversations like ChatGPT/Gemini web
"""

import os
import json
from datetime import datetime
from typing import Optional, List, Dict, Any
from contextlib import contextmanager
from urllib.parse import urlparse, unquote

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
    PSYCOPG2_AVAILABLE = True
except ImportError:
    PSYCOPG2_AVAILABLE = False
    print("Warning: psycopg2 not installed. Database features disabled.")

def parse_database_url(url: str) -> dict:
    """Parse DATABASE_URL into connection parameters using urllib"""
    if not url:
        return None
    
    try:
        parsed = urlparse(url)
        
        return {
            "host": parsed.hostname or "localhost",
            "port": parsed.port or 5432,
            "user": unquote(parsed.username or "postgres"),
            "password": unquote(parsed.password or ""),
            "database": parsed.path.lstrip("/") or "postgres"
        }
    except Exception as e:
        print(f"Error parsing DATABASE_URL: {e}")
        return None

def get_connection():
    """Get database connection"""
    if not PSYCOPG2_AVAILABLE:
        return None
    
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        return None
    
    params = parse_database_url(db_url)
    if not params:
        return None
    
    try:
        return psycopg2.connect(**params)
    except Exception as e:
        print(f"Database connection error: {e}")
        return None

@contextmanager
def get_cursor(commit=True):
    """Context manager for database cursor"""
    conn = get_connection()
    if not conn:
        yield None
        return
    
    try:
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        yield cursor
        if commit:
            conn.commit()
    except Exception as e:
        conn.rollback()
        print(f"Database error: {e}")
        raise
    finally:
        cursor.close()
        conn.close()

def init_database() -> tuple:
    """Initialize database tables"""
    if not PSYCOPG2_AVAILABLE:
        return False, "psycopg2 not installed"
    
    try:
        with get_cursor() as cursor:
            if cursor is None:
                return False, "Could not connect to database"
            
            # Create conversations table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id SERIAL PRIMARY KEY,
                    title VARCHAR(255) NOT NULL DEFAULT 'New Chat',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    document_context JSONB DEFAULT '[]',
                    tags JSONB DEFAULT '[]',
                    is_archived BOOLEAN DEFAULT FALSE,
                    metadata JSONB DEFAULT '{}'
                )
            """)
            
            # Create messages table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id SERIAL PRIMARY KEY,
                    conversation_id INTEGER REFERENCES conversations(id) ON DELETE CASCADE,
                    role VARCHAR(20) NOT NULL,
                    content TEXT NOT NULL,
                    sources JSONB DEFAULT '[]',
                    tool_calls JSONB DEFAULT '[]',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata JSONB DEFAULT '{}'
                )
            """)
            
            # Create bookmarks table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS bookmarks (
                    id SERIAL PRIMARY KEY,
                    message_id INTEGER REFERENCES messages(id) ON DELETE CASCADE,
                    note TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create indexes
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_conv_updated ON conversations(updated_at DESC)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_msg_conv ON messages(conversation_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_conv_archived ON conversations(is_archived)")
            
            return True, "Database initialized successfully"
    except Exception as e:
        return False, f"Database initialization error: {e}"

def create_conversation(title: str = "New Chat", document_context: List[str] = None) -> Optional[int]:
    """Create a new conversation"""
    try:
        with get_cursor() as cursor:
            if cursor is None:
                return None
            
            cursor.execute("""
                INSERT INTO conversations (title, document_context)
                VALUES (%s, %s)
                RETURNING id
            """, (title, json.dumps(document_context or [])))
            
            result = cursor.fetchone()
            return result['id'] if result else None
    except Exception as e:
        print(f"Error creating conversation: {e}")
        return None

def update_conversation_title(conversation_id: int, title: str) -> bool:
    """Update conversation title"""
    try:
        with get_cursor() as cursor:
            if cursor is None:
                return False
            
            cursor.execute("""
                UPDATE conversations 
                SET title = %s, updated_at = CURRENT_TIMESTAMP
                WHERE id = %s
            """, (title, conversation_id))
            
            return True
    except Exception as e:
        print(f"Error updating title: {e}")
        return False

def update_conversation_documents(conversation_id: int, documents: List[str]) -> bool:
    """Update conversation document context"""
    try:
        with get_cursor() as cursor:
            if cursor is None:
                return False
            
            cursor.execute("""
                UPDATE conversations 
                SET document_context = %s, updated_at = CURRENT_TIMESTAMP
                WHERE id = %s
            """, (json.dumps(documents), conversation_id))
            
            return True
    except Exception as e:
        print(f"Error updating documents: {e}")
        return False

def archive_conversation(conversation_id: int, archived: bool = True) -> bool:
    """Archive or unarchive a conversation"""
    try:
        with get_cursor() as cursor:
            if cursor is None:
                return False
            
            cursor.execute("""
                UPDATE conversations 
                SET is_archived = %s, updated_at = CURRENT_TIMESTAMP
                WHERE id = %s
            """, (archived, conversation_id))
            
            return True
    except Exception as e:
        print(f"Error archiving conversation: {e}")
        return False

def delete_conversation(conversation_id: int) -> bool:
    """Delete a conversation and all its messages"""
    try:
        with get_cursor() as cursor:
            if cursor is None:
                return False
            
            cursor.execute("DELETE FROM conversations WHERE id = %s", (conversation_id,))
            return True
    except Exception as e:
        print(f"Error deleting conversation: {e}")
        return False

def get_conversations(include_archived: bool = False, limit: int = 50) -> List[Dict]:
    """Get list of conversations"""
    try:
        with get_cursor(commit=False) as cursor:
            if cursor is None:
                return []
            
            if include_archived:
                cursor.execute("""
                    SELECT id, title, created_at, updated_at, document_context, tags, is_archived
                    FROM conversations
                    ORDER BY updated_at DESC
                    LIMIT %s
                """, (limit,))
            else:
                cursor.execute("""
                    SELECT id, title, created_at, updated_at, document_context, tags, is_archived
                    FROM conversations
                    WHERE is_archived = FALSE
                    ORDER BY updated_at DESC
                    LIMIT %s
                """, (limit,))
            
            results = cursor.fetchall()
            return [dict(row) for row in results]
    except Exception as e:
        print(f"Error getting conversations: {e}")
        return []

def get_conversation(conversation_id: int) -> Optional[Dict]:
    """Get a single conversation by ID"""
    try:
        with get_cursor(commit=False) as cursor:
            if cursor is None:
                return None
            
            cursor.execute("""
                SELECT id, title, created_at, updated_at, document_context, tags, is_archived, metadata
                FROM conversations
                WHERE id = %s
            """, (conversation_id,))
            
            result = cursor.fetchone()
            return dict(result) if result else None
    except Exception as e:
        print(f"Error getting conversation: {e}")
        return None

def add_message(conversation_id: int, role: str, content: str, 
                sources: List[Dict] = None, tool_calls: List[Dict] = None) -> Optional[int]:
    """Add a message to a conversation"""
    try:
        with get_cursor() as cursor:
            if cursor is None:
                return None
            
            cursor.execute("""
                INSERT INTO messages (conversation_id, role, content, sources, tool_calls)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING id
            """, (
                conversation_id, 
                role, 
                content, 
                json.dumps(sources or []),
                json.dumps(tool_calls or [])
            ))
            
            result = cursor.fetchone()
            msg_id = result['id'] if result else None
            
            # Update conversation timestamp
            cursor.execute("""
                UPDATE conversations SET updated_at = CURRENT_TIMESTAMP WHERE id = %s
            """, (conversation_id,))
            
            return msg_id
    except Exception as e:
        print(f"Error adding message: {e}")
        return None

def get_messages(conversation_id: int) -> List[Dict]:
    """Get all messages for a conversation"""
    try:
        with get_cursor(commit=False) as cursor:
            if cursor is None:
                return []
            
            cursor.execute("""
                SELECT id, role, content, sources, tool_calls, created_at
                FROM messages
                WHERE conversation_id = %s
                ORDER BY created_at ASC
            """, (conversation_id,))
            
            results = cursor.fetchall()
            messages = []
            for row in results:
                msg = dict(row)
                # Parse JSON fields
                if isinstance(msg.get('sources'), str):
                    msg['sources'] = json.loads(msg['sources'])
                if isinstance(msg.get('tool_calls'), str):
                    msg['tool_calls'] = json.loads(msg['tool_calls'])
                messages.append(msg)
            
            return messages
    except Exception as e:
        print(f"Error getting messages: {e}")
        return []

def search_conversations(query: str, limit: int = 20) -> List[Dict]:
    """Search conversations by title or message content"""
    try:
        with get_cursor(commit=False) as cursor:
            if cursor is None:
                return []
            
            search_pattern = f"%{query}%"
            cursor.execute("""
                SELECT DISTINCT c.id, c.title, c.updated_at, c.document_context
                FROM conversations c
                LEFT JOIN messages m ON c.id = m.conversation_id
                WHERE c.title ILIKE %s OR m.content ILIKE %s
                ORDER BY c.updated_at DESC
                LIMIT %s
            """, (search_pattern, search_pattern, limit))
            
            results = cursor.fetchall()
            return [dict(row) for row in results]
    except Exception as e:
        print(f"Error searching conversations: {e}")
        return []

def add_bookmark(message_id: int, note: str = None) -> Optional[int]:
    """Add a bookmark to a message"""
    try:
        with get_cursor() as cursor:
            if cursor is None:
                return None
            
            cursor.execute("""
                INSERT INTO bookmarks (message_id, note)
                VALUES (%s, %s)
                RETURNING id
            """, (message_id, note))
            
            result = cursor.fetchone()
            return result['id'] if result else None
    except Exception as e:
        print(f"Error adding bookmark: {e}")
        return None

def remove_bookmark(bookmark_id: int) -> bool:
    """Remove a bookmark"""
    try:
        with get_cursor() as cursor:
            if cursor is None:
                return False
            
            cursor.execute("DELETE FROM bookmarks WHERE id = %s", (bookmark_id,))
            return True
    except Exception as e:
        print(f"Error removing bookmark: {e}")
        return False

def get_bookmarks() -> List[Dict]:
    """Get all bookmarks with message content"""
    try:
        with get_cursor(commit=False) as cursor:
            if cursor is None:
                return []
            
            cursor.execute("""
                SELECT b.id, b.note, b.created_at, m.content, m.role, c.title as conversation_title
                FROM bookmarks b
                JOIN messages m ON b.message_id = m.id
                JOIN conversations c ON m.conversation_id = c.id
                ORDER BY b.created_at DESC
            """)
            
            results = cursor.fetchall()
            return [dict(row) for row in results]
    except Exception as e:
        print(f"Error getting bookmarks: {e}")
        return []

def add_tags_to_conversation(conversation_id: int, tags: List[str]) -> bool:
    """Add tags to a conversation"""
    try:
        with get_cursor() as cursor:
            if cursor is None:
                return False
            
            cursor.execute("""
                UPDATE conversations 
                SET tags = tags || %s, updated_at = CURRENT_TIMESTAMP
                WHERE id = %s
            """, (json.dumps(tags), conversation_id))
            
            return True
    except Exception as e:
        print(f"Error adding tags: {e}")
        return False

def get_conversations_by_tag(tag: str) -> List[Dict]:
    """Get conversations with a specific tag"""
    try:
        with get_cursor(commit=False) as cursor:
            if cursor is None:
                return []
            
            cursor.execute("""
                SELECT id, title, created_at, updated_at, document_context, tags
                FROM conversations
                WHERE tags ? %s
                ORDER BY updated_at DESC
            """, (tag,))
            
            results = cursor.fetchall()
            return [dict(row) for row in results]
    except Exception as e:
        print(f"Error getting conversations by tag: {e}")
        return []

def export_conversation(conversation_id: int, format: str = "markdown") -> Optional[str]:
    """Export a conversation to markdown or JSON format"""
    try:
        conv = get_conversation(conversation_id)
        if not conv:
            return None
        
        messages = get_messages(conversation_id)
        
        if format == "json":
            export_data = {
                "conversation": conv,
                "messages": messages,
                "exported_at": datetime.now().isoformat()
            }
            return json.dumps(export_data, indent=2, default=str)
        
        # Markdown format
        md = f"# {conv['title']}\n\n"
        md += f"**Created:** {conv['created_at']}\n"
        md += f"**Updated:** {conv['updated_at']}\n"
        
        if conv.get('document_context'):
            md += f"**Documents:** {', '.join(conv['document_context'])}\n\n"
        
        md += "---\n\n"
        
        for msg in messages:
            role_icon = "[User]" if msg['role'] == "user" else "[Assistant]"
            md += f"### {role_icon} {msg['role'].capitalize()}\n\n"
            md += f"{msg['content']}\n\n"
            
            if msg.get('sources'):
                md += "<details>\n<summary>Sources</summary>\n\n"
                for src in msg['sources']:
                    md += f"- {src.get('source', 'Unknown')} (Page {src.get('page', 'N/A')})\n"
                md += "\n</details>\n\n"
            
            md += "---\n\n"
        
        return md
    except Exception as e:
        print(f"Error exporting conversation: {e}")
        return None

def generate_title_from_message(content: str, max_length: int = 50) -> str:
    """Generate a title from the first message content"""
    title = content.strip()
    
    if '\n' in title:
        title = title.split('\n')[0]
    if '.' in title and len(title.split('.')[0]) < max_length:
        title = title.split('.')[0]
    
    if len(title) > max_length:
        title = title[:max_length-3] + "..."
    
    return title or "New Chat"

def test_connection() -> tuple:
    """Test database connection"""
    if not PSYCOPG2_AVAILABLE:
        return False, "psycopg2 not installed"
    
    try:
        conn = get_connection()
        if conn:
            conn.close()
            return True, "Connection successful"
        return False, "Could not establish connection"
    except Exception as e:
        return False, f"Connection error: {e}"
