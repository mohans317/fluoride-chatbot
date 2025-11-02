import streamlit as st
import requests
from sentence_transformers import SentenceTransformer
import chromadb

st.set_page_config(page_title="Fluoride Chatbot", page_icon="💧", layout="wide")

st.markdown("<h1 style='text-align: center; color: #1f77b4;'>💧 Fluoride Chatbot</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #666;'>Answers based on your provided fluoride data</p>", unsafe_allow_html=True)

# Get API key
try:
    api_key = st.secrets["PERPLEXITY_API_KEY"]
    api_key_loaded = True
except KeyError:
    api_key_loaded = False

# Initialize embeddings and ChromaDB
@st.cache_resource
def load_chroma():
    embeddings = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_collection(name="langchain")  # CORRECT collection name
    return embeddings, collection

try:
    embeddings, collection = load_chroma()
    chroma_loaded = True
except Exception as e:
    st.error(f"❌ Error loading ChromaDB: {str(e)}")
    chroma_loaded = False

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    if api_key_loaded:
        st.markdown('<div style="background-color: #d4edda; color: #155724; padding: 10px; border-radius: 5px;">✅ API Key loaded from Secrets</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div style="background-color: #f8d7da; color: #721c24; padding: 10px; border-radius: 5px;">❌ API Key not found</div>', unsafe_allow_html=True)
    
    if chroma_loaded:
        st.markdown('<div style="background-color: #d4edda; color: #155724; padding: 10px; border-radius: 5px;">✅ Knowledge base loaded (1,529 documents)</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div style="background-color: #f8d7da; color: #721c24; padding: 10px; border-radius: 5px;">❌ Knowledge base not found</div>', unsafe_allow_html=True)
    
    model_name = st.selectbox(
        "🤖 Select Model",
        ["sonar", "sonar-reasoning", "sonar-pro"]
    )
    
    st.markdown("---")
    st.markdown("""
    ### 📚 About This Chatbot
    - ✅ Uses YOUR fluoride data (1,529 documents)
    - ✅ Shows sources for each answer
    - ✅ Simple language for everyone
    - ✅ Secure API key handling
    - ✅ Powered by Perplexity Sonar AI
    """)

if not api_key_loaded or not chroma_loaded:
    st.error("⚠️ Please check configuration in sidebar!")
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
                # Step 1: Search ChromaDB for relevant documents
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
                    # Step 2: Format context from YOUR documents
                    context = "\n\n---\n\n".join(retrieved_docs)
                    
                    # Step 3: Send to Perplexity with instruction to use ONLY this data
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
1. Answer ONLY using the provided data below - do NOT use external knowledge
2. If the answer is NOT in the data, say: "I don't have this information in my knowledge base"
3. Use simple, easy-to-understand language for everyone
4. Avoid technical jargon - explain in simple terms
5. If you use technical terms, explain what they mean
6. Always be accurate and factual

DATA FROM KNOWLEDGE BASE:
{}""".format(context)
                            },
                            {
                                "role": "user",
                                "content": prompt
                            }
                        ],
                        "max_tokens": 2048,
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
                        
                        # Show sources
                        with st.expander("📚 Sources Used"):
                            for i, source in enumerate(retrieved_docs, 1):
                                st.markdown(f"**Source {i}:**")
                                st.text(source[:300] + "..." if len(source) > 300 else source)
                    else:
                        error_response = response.json() if response.headers.get('content-type') == 'application/json' else response.text
                        st.error(f"❌ Error {response.status_code}")
                        
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")


