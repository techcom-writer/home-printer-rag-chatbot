import streamlit as st
from groq import Groq
import pinecone
from pinecone import Pinecone
import requests
import os
from typing import Optional
import json

# Configure Streamlit
st.set_page_config(
    page_title="Home Printer Documentation Bot",
    page_icon="🖨️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Hide Streamlit elements when embedded
st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# ============= CONFIGURATION =============
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", os.environ.get("GROQ_API_KEY"))
PINECONE_API_KEY = st.secrets.get("PINECONE_API_KEY", os.environ.get("PINECONE_API_KEY"))
PINECONE_INDEX = st.secrets.get("PINECONE_INDEX", "home-printer-docs")
HUGGINGFACE_API_KEY = st.secrets.get("HUGGINGFACE_API_KEY", os.environ.get("HUGGINGFACE_API_KEY"))

# Hugging Face Embedding Model
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
HF_EMBED_URL = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{EMBEDDING_MODEL}"

# ============= HELPER FUNCTIONS =============

def get_embedding(text: str) -> Optional[list]:
    """Get embeddings from Hugging Face."""
    headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}
    response = requests.post(
        HF_EMBED_URL,
        headers=headers,
        json={"inputs": text},
        timeout=30
    )
    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"Embedding error: {response.text}")
        return None

def retrieve_context(query: str, top_k: int = 3) -> str:
    """Retrieve relevant context from Pinecone."""
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        index = pc.Index(PINECONE_INDEX)
        
        # Get query embedding
        query_embedding = get_embedding(query)
        if query_embedding is None:
            return "Unable to process query embeddings."
        
        # Query Pinecone
        results = index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        
        # Build context from results
        context_parts = []
        for match in results.get("matches", []):
            if "metadata" in match and "text" in match["metadata"]:
                context_parts.append(match["metadata"]["text"])
        
        return "\n".join(context_parts) if context_parts else "No relevant documentation found."
    
    except Exception as e:
        st.error(f"Retrieval error: {str(e)}")
        return "Error retrieving context."

def generate_response(system_prompt: str, user_query: str) -> str:
    """Generate response using Groq LLM."""
    try:
        client = Groq(api_key=GROQ_API_KEY)
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query}
        ]
        
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",  # Fast and free-tier friendly
            messages=messages,
            temperature=0.7,
            max_tokens=1024
        )
        
        return response.choices[0].message.content
    
    except Exception as e:
        return f"Error generating response: {str(e)}"

# ============= UI LAYOUT =============

# Header
col1, col2 = st.columns([0.1, 0.9])
with col1:
    st.markdown("🖨️")
with col2:
    st.title("Home Printer Documentation Bot")

st.markdown("Ask questions about home printer setup, maintenance, and troubleshooting.")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if user_query := st.chat_input("Ask about printer setup, configuration, or troubleshooting..."):
    
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": user_query})
    
    with st.chat_message("user"):
        st.markdown(user_query)
    
    # Show loading state
    with st.chat_message("assistant"):
        with st.spinner("Searching documentation..."):
            
            # Step 1: Retrieve context from vector DB
            context = retrieve_context(user_query, top_k=3)
            
            # Step 2: Build system prompt
            system_prompt = f"""You are a helpful Home Printer Support Assistant. 
Answer the user's question ONLY based on the following documentation context. 
Use clear, concise language. Format your response with paragraphs or bullet points.
Do NOT output code blocks, tables, or markdown formatting beyond basic structure.

DOCUMENTATION CONTEXT:
{context}

If the context doesn't contain relevant information, politely explain that the question 
is outside your knowledge base and suggest checking the full documentation."""
            
            # Step 3: Generate response
            bot_response = generate_response(system_prompt, user_query)
            
            st.markdown(bot_response)
            
            # Add assistant message to history
            st.session_state.messages.append({"role": "assistant", "content": bot_response})

# ============= SIDEBAR INFO =============
with st.sidebar:
    st.subheader("📖 About This Bot")
    st.info("""
    This chatbot uses:
    - **Groq**: Fast LLM inference
    - **Pinecone**: Vector database for documentation
    - **Hugging Face**: Embeddings
    
    It retrieves relevant documentation and generates accurate, context-aware answers.
    """)
    
    st.subheader("🔍 Sample Questions")
    st.markdown("""
    - How do I install a home printer?
    - What are the recommended print settings?
    - How do I troubleshoot offline printer issues?
    - How often should I maintain my printer?
    - What causes poor print quality?
    """)
    
    st.subheader("📚 Full Documentation")
    st.markdown("""
    For comprehensive guides, visit the full 
    [Home Printer Documentation](https://your-username.github.io/home-printer-docs/)
    """)
    
    # Debug info (hidden in production)
    if st.secrets.get("DEBUG_MODE", False):
        st.divider()
        st.subheader("🐛 Debug Info")
        st.caption(f"Pinecone Index: {PINECONE_INDEX}")
        st.caption(f"Embedding Model: {EMBEDDING_MODEL}")
