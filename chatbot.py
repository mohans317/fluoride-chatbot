import os
import streamlit as st

# ------ Robust Utility Functions ------

def safe_get_env(name, secret_fallback=None, error_if_missing=True):
    # Prefer environment, fallback to secrets (for local Streamlit Cloud), else throw error or return None
    val = os.environ.get(name)
    if val: return val
    try:
        import streamlit as _st
        if hasattr(_st, "secrets"):
            val = _st.secrets.get(name)
    except Exception:
        pass
    if (not val) and error_if_missing:
        st.error(f"❌ Missing critical configuration: '{name}'. Please set it in cloud dashboard Environment (or .streamlit/secrets.toml for local debug)")
        st.stop()
    return val or secret_fallback

# ------ Application Setup ------

st.set_page_config(page_title="Fluoride Chatbot", page_icon="💧", layout="wide")
st.title("💧 Fluoride Chatbot")
st.write("Answers based on your provided fluoride data. Robust, cloud-first template.")

# API Key load
API_KEY = safe_get_env("PERPLEXITY_API_KEY")

# ChromaDB Setup (Cloud Safe):
def load_chroma_db(db_dir="./chroma_db"):
    db_path = os.path.abspath(db_dir)
    if not os.path.exists(db_path):
        st.warning(f"Chroma DB directory not found: {db_path}\nIf deploying to cloud, upload your chroma_db/ to your repo or switch to a managed cloud vector db.")
        return None, None
    try:
        from sentence_transformers import SentenceTransformer
        import chromadb
        embeddings = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        client = chromadb.PersistentClient(path=db_path)
        colnames = [c.name for c in client.list_collections()]
        collection = client.get_collection(name=colnames[0]) if colnames else None
        if not collection:
            st.warning("No collection found in chroma_db.")
        return embeddings, collection
    except Exception as e:
        st.error(f"Failed to initialize vector DB: {str(e)}")
        return None, None

embeddings, collection = load_chroma_db()

# ------ Sidebar Diagnostics ------
with st.sidebar:
    st.header("Diagnostics")
    st.markdown("**API Key:** " + ("✅ Found" if API_KEY else "❌ Missing"))
    st.markdown("**DB:** " + ("✅ Loaded" if collection else "❌ Not Found"))
    if collection:
        try:
            count = collection.count()
            st.markdown(f"**Docs in DB:** {count}")
        except: st.markdown("**Docs in DB:** ❌ Error")

# ------ Main Chat Loop ------

if not API_KEY or not collection:
    st.info("Please check setup before asking questions.")
    st.stop()

if "messages" not in st.session_state:
    st.session_state["messages"] = []

for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

prompt = st.chat_input("Ask about fluoride in drinking water...")

if prompt:
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Retrieving..."):
            try:
                # --- Embeddings & Search ---
                query_vector = embeddings.encode([prompt])[0].tolist()
                res = collection.query(query_embeddings=[query_vector], n_results=3, include=['documents'])
                docs = res['documents'][0] if (res and res.get("documents")) else []
                if not docs:
                    answer = "❗I couldn't find relevant info in your knowledge base."
                    st.markdown(answer)
                    st.session_state["messages"].append({"role": "assistant", "content": answer})
                else:
                    import requests
                    context = "\n\n".join(docs)
                    SYSTEM_PROMPT = ("You are an assistant answering about fluoride in water. "
                                     "Use ONLY the provided facts below.")
                    payload = {
                        "model": "sonar",  # update as needed
                        "messages": [
                            {"role": "system", "content": SYSTEM_PROMPT + "\n" + context},
                            {"role": "user", "content": prompt}
                        ],
                        "max_tokens": 1000, "temperature": 0.2
                    }
                    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
                    r = requests.post("https://api.perplexity.ai/chat/completions", headers=headers, json=payload, timeout=45)
                    if r.status_code == 200:
                        out = r.json()["choices"][0]["message"]["content"]
                        st.markdown(out)
                        st.session_state["messages"].append({"role": "assistant", "content": out})
                    else:
                        err = f"API error: {r.status_code} {r.text[:100]}"
                        st.error(err)
                        st.session_state["messages"].append({"role": "assistant", "content": err})
            except Exception as ex:
                st.error("Unexpected error: " + str(ex))
                st.session_state





