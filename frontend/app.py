import streamlit as st
import requests
from datetime import datetime
import uuid

# Configuration
API_BASE_URL = "http://localhost:8000/api/v1"  # Changed from backend:8000 to localhost:8000

# Page config
st.set_page_config(
    page_title="Multimodal RAG System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        color: #1a1a1a;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
        color: #0d47a1;
    }
    .assistant-message {
        background-color: #f5f5f5;
        border-left: 4px solid #4caf50;
        color: #1b5e20;
    }
    .sidebar-thread {
        padding: 0.75rem;
        border-radius: 0.5rem;
        margin-bottom: 0.5rem;
        cursor: pointer;
        transition: background-color 0.2s;
    }
    .sidebar-thread:hover {
        background-color: #e0e0e0;
    }
    .sidebar-thread-active {
        background-color: #bbdefb;
        border-left: 4px solid #1976d2;
    }
    .document-card {
        background-color: #fff3e0;
        padding: 0.75rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ff9800;
        margin-bottom: 0.5rem;
    }
    .danger-zone {
        background-color: #ffebee;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #f44336;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize session state variables."""
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = None
    if "conversations" not in st.session_state:
        st.session_state.conversations = []
    if "current_messages" not in st.session_state:
        st.session_state.current_messages = []
    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = []
    if "show_reset_confirm" not in st.session_state:
        st.session_state.show_reset_confirm = False
    if "reset_confirmation_text" not in st.session_state:
        st.session_state.reset_confirmation_text = ""


def fetch_conversations():
    """Fetch all conversation threads from API."""
    try:
        response = requests.get(f"{API_BASE_URL}/conversations", timeout=5)
        if response.status_code == 200:
            st.session_state.conversations = response.json()
        else:
            st.warning("No conversations found yet")
    except Exception as e:
        st.warning(f"Could not fetch conversations: {e}")


def load_conversation(thread_id: str):
    """Load a specific conversation."""
    try:
        response = requests.get(f"{API_BASE_URL}/conversations/{thread_id}", timeout=5)
        if response.status_code == 200:
            data = response.json()
            st.session_state.thread_id = thread_id
            st.session_state.current_messages = data["messages"]
            st.experimental_rerun()
    except Exception as e:
        st.error(f"Error loading conversation: {e}")


def delete_conversation(thread_id: str):
    """Delete a conversation thread."""
    try:
        response = requests.delete(f"{API_BASE_URL}/conversations/{thread_id}", timeout=5)
        if response.status_code == 200:
            if st.session_state.thread_id == thread_id:
                st.session_state.thread_id = None
                st.session_state.current_messages = []
            fetch_conversations()
            st.experimental_rerun()
    except Exception as e:
        st.error(f"Error deleting conversation: {e}")


def new_conversation():
    """Start a new conversation."""
    st.session_state.thread_id = None
    st.session_state.current_messages = []
    st.experimental_rerun()


def reset_database():
    """Reset the entire database (vector store and conversations)."""
    try:
        response = requests.post(f"{API_BASE_URL}/reset", timeout=10)
        if response.status_code == 200:
            # Clear local session state
            st.session_state.thread_id = None
            st.session_state.current_messages = []
            st.session_state.conversations = []
            st.session_state.uploaded_files = []
            st.session_state.show_reset_confirm = False
            st.session_state.reset_confirmation_text = ""
            st.success("✅ Database reset successfully! All documents and conversations have been deleted.")
            st.experimental_rerun()
        else:
            st.error(f"Reset failed: {response.text}")
    except Exception as e:
        st.error(f"Error resetting database: {e}")


def send_query(query: str, include_images: bool = True):
    """Send query to API."""
    try:
        payload = {
            "query": query,
            "thread_id": st.session_state.thread_id,
            "include_images": include_images,
            "top_k": 5
        }
        
        response = requests.post(f"{API_BASE_URL}/query", json=payload, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            
            # Update thread_id if it's a new conversation
            if not st.session_state.thread_id:
                st.session_state.thread_id = data.get("thread_id")
            
            # Add messages to current conversation
            st.session_state.current_messages.append({
                "role": "user",
                "content": query,
                "timestamp": datetime.now().isoformat()
            })
            st.session_state.current_messages.append({
                "role": "assistant",
                "content": data["answer"],
                "timestamp": datetime.now().isoformat(),
                "retrieved_documents": data.get("retrieved_documents", []),
                "processing_time": data.get("processing_time", 0)
            })
            
            # Refresh conversations list
            fetch_conversations()
            
            return data
        else:
            st.error(f"Query failed: {response.text}")
            return None
    except Exception as e:
        st.error(f"Error sending query: {e}")
        return None


def upload_documents(files):
    """Upload documents to API."""
    try:
        files_data = [("files", (file.name, file, file.type)) for file in files]
        response = requests.post(f"{API_BASE_URL}/upload/batch", files=files_data, timeout=60)
        
        if response.status_code == 200:
            data = response.json()
            return data
        else:
            st.error(f"Upload failed: {response.text}")
            return None
    except Exception as e:
        st.error(f"Error uploading documents: {e}")
        return None


def main():
    """Main application."""
    initialize_session_state()
    
    # Fetch conversations on load
    if not st.session_state.conversations:
        fetch_conversations()
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 💬 Conversations")
        
        if st.button("➕ New Chat", use_container_width=True):
            new_conversation()
        
        st.markdown("---")
        
        # Display conversation threads
        if st.session_state.conversations:
            for conv in st.session_state.conversations:
                is_active = conv["thread_id"] == st.session_state.thread_id
                
                col1, col2 = st.columns([4, 1])
                
                with col1:
                    if st.button(
                        conv["title"],
                        key=f"conv_{conv['thread_id']}",
                        use_container_width=True,
                        type="primary" if is_active else "secondary"
                    ):
                        load_conversation(conv["thread_id"])
                
                with col2:
                    if st.button("🗑️", key=f"del_{conv['thread_id']}"):
                        delete_conversation(conv["thread_id"])
        else:
            st.info("No conversations yet. Start a new chat!")
        
        st.markdown("---")
        
        # Document upload section
        st.markdown("### 📁 Upload Documents")
        uploaded_files = st.file_uploader(
            "Upload files",
            type=["txt", "md", "csv", "pdf", "png", "jpg", "jpeg"],
            accept_multiple_files=True,
            key="file_uploader"
        )
        
        if uploaded_files and st.button("Upload", use_container_width=True):
            with st.spinner("Uploading documents..."):
                result = upload_documents(uploaded_files)
                if result:
                    st.success(f"Uploaded {len(uploaded_files)} files successfully!")
                    st.session_state.uploaded_files.extend(uploaded_files)
        
        st.markdown("---")
        
        # Database Reset Section (Danger Zone)
        st.markdown("### ⚠️ Danger Zone")
        
        if not st.session_state.show_reset_confirm:
            if st.button("🗑️ Reset Database", use_container_width=True, type="secondary"):
                st.session_state.show_reset_confirm = True
                st.experimental_rerun()
        else:
            st.warning("⚠️ **WARNING**: This will delete ALL documents and conversations permanently!")
            st.markdown("Type **DELETE** to confirm:")
            
            confirmation = st.text_input(
                "Confirmation",
                key="reset_confirm_input",
                placeholder="Type DELETE to confirm",
                label_visibility="collapsed"
            )
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Cancel", use_container_width=True):
                    st.session_state.show_reset_confirm = False
                    st.session_state.reset_confirmation_text = ""
                    st.experimental_rerun()
            with col2:
                if st.button("Confirm Reset", use_container_width=True, type="primary", disabled=(confirmation != "DELETE")):
                    if confirmation == "DELETE":
                        reset_database()
    
    # Main content area
    st.markdown("<h1 class='main-header'>🤖 Multimodal RAG System</h1>", unsafe_allow_html=True)
    
    # Display current conversation
    chat_container = st.container()
    
    with chat_container:
        if not st.session_state.current_messages:
            st.info("👋 Welcome! Upload some documents and start asking questions.")
        
        for msg in st.session_state.current_messages:
            if msg["role"] == "user":
                st.markdown(
                    f"<div class='chat-message user-message'><b>You:</b><br>{msg['content']}</div>",
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"<div class='chat-message assistant-message'><b>Assistant:</b><br>{msg['content']}</div>",
                    unsafe_allow_html=True
                )
                
                # Show retrieved documents if available
                if "retrieved_documents" in msg and msg["retrieved_documents"]:
                    with st.expander(f"📚 Retrieved Documents ({len(msg['retrieved_documents'])})"):
                        for i, doc in enumerate(msg["retrieved_documents"], 1):
                            # Handle both dict and object formats
                            if isinstance(doc, dict):
                                file_name = doc.get('metadata', {}).get('file_name', 'Unknown')
                                relevance_score = doc.get('relevance_score', 0)
                                content = doc.get('content', '')
                            else:
                                file_name = doc.metadata.get('file_name', 'Unknown')
                                relevance_score = doc.relevance_score
                                content = doc.content
                            
                            st.markdown(
                                f"""
                                <div class='document-card'>
                                    <b>Document {i}</b> - {file_name}<br>
                                    <small>Relevance: {relevance_score:.2%}</small><br>
                                    {content[:200]}...
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                
                # Show processing time
                if "processing_time" in msg:
                    st.caption(f"⏱️ Processed in {msg['processing_time']:.2f}s")
    
    # Query input at the bottom
    st.markdown("---")
    
    col1, col2 = st.columns([4, 1])
    
    with col1:
        query = st.text_input(
            "Ask a question...",
            key="query_input",
            placeholder="What would you like to know?"
        )
    
    with col2:
        include_images = st.checkbox("Include Images", value=True)
    
    if st.button("Send", use_container_width=True) and query:
        with st.spinner("Processing..."):
            send_query(query, include_images)
            st.experimental_rerun()
    
    # Show system status
    with st.expander("ℹ️ System Status"):
        try:
            response = requests.get(f"{API_BASE_URL}/health", timeout=5)
            if response.status_code == 200:
                health = response.json()
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Status", health["status"])
                with col2:
                    st.metric("Vector Store", health["vector_store_status"])
                with col3:
                    st.metric("Total Documents", health["total_documents"])
        except Exception as e:
            st.error(f"Could not fetch system status: {e}")


if __name__ == "__main__":
    main()