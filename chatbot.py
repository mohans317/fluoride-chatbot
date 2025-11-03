import os
import sys
import streamlit as st
import requests
from sentence_transformers import SentenceTransformer
import chromadb

st.set_page_config(page_title="Fluoride Chatbot", page_icon="💧", layout="wide")

st.markdown("<h1 style='text-align: center; color: #1f77b4;'>💧 Fluoride Chatbot</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #666;'>Answers based on your provided fluoride data</p>", unsafe_allow_html=True)

# Get API key from environment
api_key = os.environ.get("PERPLEXITY_API_KEY")

if not api_key:
    st.error("❌ ERROR: PERPLEXITY_API_KEY environment variable is not set!")
    st.info("Please add PERPLEXITY_API_KEY to your Render environment variables in Settings")
    st.stop()

# Initialize embeddings and ChromaDB with better error handling
@st.cache_resource
def load_chroma():
    try:
        # Try to load with absolute path for Render
        db_path = os.path.abspath("./chroma_db")
        
        if not os.path.exists(db_path):
            st.warning(f"⚠️ Database path not found: {db_path}")
            return None, None
        
        embeddings = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        client = chromadb.PersistentClient(path=db_path)
        
        try:
            collection = client.get_collection(name="langchain")
        except:
            # Try alternative collection names
            collections = client.list_collections()
            if collections:
                collection = client.get_collection(name=collections[0].name)
            else:
                return embeddings, None
        
        return embeddings, collection
    except Exception as e:
        st.error(f"❌ Error loading ChromaDB: {str(e)}")
        return None, None

embeddings, collection = load_chroma()

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    if api_key:
        st.markdown('<div style="background-color: #d4edda; color: #155724; padding: 10px; border-radius: 5px;">✅ API Key loaded</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div style="background-color: #f8d7da; color: #721c24; padding: 10px; border-radius: 5px;">❌ API Key not configured</div>', unsafe_allow_html=True)
    
    if collection:
        try:
            count = collection.count()
            st.markdown(f'<div style="background-color: #d4edda; color: #155724; padding: 10px; border-radius: 5px;">✅ Knowledge base loaded ({count} documents)</div>', unsafe_allow_html=True)
        except:
            st.markdown('<div style="background-color: #f8d7da; color: #721c24; padding: 10px; border-radius: 5px;">❌ Knowledge base error</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div style="background-color: #f8d7da; color: #721c24; padding: 10px; border-radius: 5px;">❌ Knowledge base not found</div>', unsafe_allow_html=True)
    
    model_name = st.selectbox(
        "🤖 Select Model",
        ["sonar", "sonar-reasoning", "sonar-pro"]
    )
    
    st.markdown("---")
    st.markdown("""
    ### 📚 About This Chatbot
    - ✅ Uses YOUR fluoride data
    - ✅ Shows sources for each answer
    - ✅ Simple language for everyone
    - ✅ Powered by Perplexity Sonar AI
    """)

if not embeddings or not collection:
    st.error("⚠️ Knowledge base not available. Please check ChromaDB files are in the repository.")
    st.stop()

# Chat interface
st.header("Chat")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander("📚 Sources Used"):
                for i, source in enumerate(message["sources"], 1):
                    st.markdown(f"**Source {i}:**")
                    st.text(source[:300] + "..." if len(source) > 300 else source)

# Chat input
if prompt := st.chat_input("Ask about fluoride in drinking water..."):
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("🔍 Searching your fluoride data..."):
            try:
                # Step 1: Search ChromaDB
                query_embedding = embeddings.encode([prompt])[0].tolist()
                results = collection.query(
                    query_embeddings=[query_embedding],
                    n_results=3,
                    include=['documents']
                )
                
                retrieved_docs = results['documents'][0] if results['documents'] else []
                
                if not retrieved_docs:
                    answer = "⚠️ Sorry, I couldn't find relevant information about your question in the knowledge base."
                    st.markdown(answer)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer
                    })
                else:
                    # Step 2: Format context
                    context = "\n\n---\n\n".join(retrieved_docs)
                    
                    # Step 3: Call Perplexity API
                    url = "https://api.perplexity.ai/chat/completions"
                    
                    headers = {
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json"
                    }
                    
                    payload = {
                        "model": model_name,
                        "messages": [
                            {
                                "role": "system",
                                "content": """You are a helpful assistant answering questions about fluoride in drinking water.

IMPORTANT RULES:
1. Answer ONLY using the provided data
2. If the answer is NOT in the data, say: "I don't have this information"
3. Use simple, easy-to-understand language
4. Avoid technical jargon
5. Be accurate and factual

DATA FROM KNOWLEDGE BASE:
{}""".format(context)
                            },
                            {
                                "role": "user",
                                "content": prompt
                            }
                        ],
                        "max_tokens": 1024,
                        "temperature": 0.2
                    }
                    
                    response = requests.post(url, headers=headers, json=payload, timeout=30)
                    
                    if response.status_code == 200:
                        data = response.json()
                        answer = data["choices"][0]["message"]["content"]
                        st.markdown(answer)
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": answer,
                            "sources": retrieved_docs
                        })
                        
                        with st.expander("📚 Sources Used"):
                            for i, source in enumerate(retrieved_docs, 1):
                                st.markdown(f"**Source {i}:**")
                                st.text(source[:300] + "..." if len(source) > 300 else source)
                    else:
                        st.error(f"❌ API Error {response.status_code}: {response.text}")
                        
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")




