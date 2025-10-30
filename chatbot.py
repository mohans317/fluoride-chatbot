import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
import os

# Page configuration
st.set_page_config(
    page_title="Fluoride Chatbot",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
    <style>
    .main-header {
        text-align: center;
        color: #1f77b4;
        margin-bottom: 10px;
    }
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 16px;
        margin-bottom: 30px;
    }
    .status-success {
        background-color: #d4edda;
        color: #155724;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .status-error {
        background-color: #f8d7da;
        color: #721c24;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)


class RAGChatbot:
    """
    Retrieval Augmented Generation Chatbot using Gemini 2.5 Pro and ChromaDB
    """
    
    def __init__(self, persist_directory='./chroma_db'):
        """
        Initialize the RAG chatbot
        
        Args:
            persist_directory: Path to ChromaDB vector store
        """
        self.persist_directory = persist_directory
        self.vectorstore = None
        self.chain = None
        self.embeddings = None
        self.retriever = None
        self.llm = None
        
    def initialize_embeddings(self):
        """
        Initialize HuggingFace embeddings (FREE - no API cost)
        
        Returns:
            Boolean indicating success
        """
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'}
            )
            return True
        except Exception as e:
            st.error(f"❌ Error initializing embeddings: {str(e)}")
            return False
    
    def load_vectorstore(self):
        """
        Load vector database from local storage
        
        Returns:
            Tuple of (success_boolean, doc_count)
        """
        try:
            if not self.embeddings:
                self.initialize_embeddings()
            
            self.vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )
            
            # Get collection count
            collection = self.vectorstore.get()
            doc_count = len(collection['ids']) if collection['ids'] else 0
            
            return True, doc_count
            
        except Exception as e:
            st.error(f"❌ Error loading vector database: {str(e)}")
            return False, 0
    
    def create_chain(self, api_key, model_name="gemini-2.5-pro"):
        """
        Create RAG chain with Google Gemini 2.5 Pro
        
        Args:
            api_key: Google Gemini API key
            model_name: Model to use (gemini-2.5-pro or gemini-2.5-pro-exp-05-21)
            
        Returns:
            Boolean indicating success
        """
        
        try:
            # Initialize Google Gemini LLM
            self.llm = ChatGoogleGenerativeAI(
                model=model_name,
                temperature=0,
                google_api_key=api_key
            )
            
            # Create retriever
            self.retriever = self.vectorstore.as_retriever(
                search_kwargs={"k": 3}
            )
            
            return True
            
        except Exception as e:
            st.error(f"❌ Error creating chain: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def get_response(self, question):
        """
        Get response from chatbot for a given question
        
        Args:
            question: User's question
            
        Returns:
            Tuple of (response_text, source_documents)
        """
        try:
            # Retrieve relevant documents
            docs = self.retriever.invoke(question)
            
            # Format documents as context
            context = "\n\n".join([doc.page_content for doc in docs])
            
            # Create prompt
            template = """You are a helpful and knowledgeable assistant that answers questions based ONLY on the provided context about fluoride in drinking water.

IMPORTANT RULES:
1. Answer ONLY using information from the provided context
2. If the answer is not in the context, say: "I don't have enough information in the provided sources to answer that question."
3. Never use external knowledge or make up information
4. Always be factual and accurate
5. If relevant, mention which parts of the context support your answer

Context: {context}

Question: {question}

Answer:"""
            
            prompt = PromptTemplate(
                template=template,
                input_variables=["context", "question"]
            )
            
            # Format the prompt
            formatted_prompt = prompt.format(context=context, question=question)
            
            # Get response from LLM
            response = self.llm.invoke(formatted_prompt)
            
            # Extract text from response
            if hasattr(response, 'content'):
                response_text = response.content
            else:
                response_text = str(response)
            
            return response_text, docs
            
        except Exception as e:
            error_msg = f"❌ Error generating response: {str(e)}"
            st.error(error_msg)
            import traceback
            traceback.print_exc()
            return error_msg, []


def get_api_key_from_secrets():
    """
    Securely get API key from Streamlit Secrets
    
    Returns:
        API key string or None
    """
    try:
        if "GOOGLE_API_KEY" in st.secrets:
            return st.secrets["GOOGLE_API_KEY"]
    except:
        pass
    return None


def main():
    """Main Streamlit application"""
    
    # Main header
    st.markdown("<h1 class='main-header'>💧 Fluoride Chatbot</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Ask questions about fluoride in drinking water</p>", unsafe_allow_html=True)
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Get API key from Streamlit Secrets
        api_key = get_api_key_from_secrets()
        
        if api_key:
            st.markdown('<div class="status-success">✅ API Key loaded from Streamlit Secrets</div>', 
                       unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-error">❌ API Key not found in Streamlit Secrets</div>', 
                       unsafe_allow_html=True)
            st.warning("Please add GOOGLE_API_KEY to .streamlit/secrets.toml")
        
        # Model selection - Updated with Gemini 2.5 Pro models
        model_name = st.selectbox(
            "🤖 Select Gemini Model",
            ["gemini-2.5-pro", "gemini-2.5-pro-exp-05-21"],
            help="gemini-2.5-pro: Latest stable | gemini-2.5-pro-exp-05-21: Experimental version"
        )
        
        st.markdown("---")
        
        # Information section
        st.markdown("""
        ### 📚 About This Chatbot
        
        **Features:**
        - ✅ Answers based on provided data only
        - ✅ Shows source documents for each answer
        - ✅ Free with Google Gemini API
        - ✅ Secure API key handling
        - ✅ Powered by Gemini 2.5 Pro (Latest & Most Powerful)
        
        **How it works:**
        1. Your question is converted to embeddings
        2. Similar content is retrieved from the knowledge base
        3. Gemini 2.5 Pro generates answer based on retrieved content
        
        **Security:**
        - 🔒 API key stored in Streamlit Secrets
        - 🔒 Never logged or exposed
        - 🔒 Data not stored permanently
        """)
        
        st.markdown("---")
        st.caption("Built with Streamlit, LangChain, and Google Gemini 2.5 Pro")
    
    # Initialize session state variables
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = RAGChatbot()
    
    if 'vectorstore_loaded' not in st.session_state:
        st.session_state.vectorstore_loaded = False
    
    if 'chain_ready' not in st.session_state:
        st.session_state.chain_ready = False
    
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    if 'doc_count' not in st.session_state:
        st.session_state.doc_count = 0
    
    if 'current_model' not in st.session_state:
        st.session_state.current_model = "gemini-2.5-pro"
    
    # ========== INITIALIZATION STEPS ==========
    
    # Step 1: Load vector database
    if not st.session_state.vectorstore_loaded:
        with st.spinner("📚 Loading knowledge base..."):
            success, doc_count = st.session_state.chatbot.load_vectorstore()
            if success:
                st.session_state.vectorstore_loaded = True
                st.session_state.doc_count = doc_count
                st.success(f"✅ Knowledge base loaded! ({doc_count} documents)")
            else:
                st.error("❌ Failed to load knowledge base. Please check if chroma_db folder exists.")
                st.stop()
    
    # Step 2: Check for API key
    if not api_key:
        st.error("❌ Google Gemini API key not found!")
        st.stop()
    
    # Step 3: Create chain if not ready or model changed
    if not st.session_state.chain_ready or st.session_state.current_model != model_name:
        with st.spinner("🔧 Initializing chatbot with Gemini 2.5 Pro..."):
            if st.session_state.chatbot.create_chain(api_key, model_name):
                st.session_state.chain_ready = True
                st.session_state.current_model = model_name
                st.success("✅ Chatbot ready! Using Gemini 2.5 Pro")
            else:
                st.error("Failed to initialize chatbot")
                st.stop()
    
    # ========== CHAT INTERFACE ==========
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Show sources if available
            if "sources" in message and message["sources"]:
                with st.expander("📚 View Sources"):
                    for i, source in enumerate(message["sources"], 1):
                        st.markdown(f"**Source {i}:**")
                        # Display content preview
                        content_preview = source.page_content[:500]
                        if len(source.page_content) > 500:
                            content_preview += "..."
                        st.text(content_preview)
                        
                        # Display metadata
                        if 'chunk_id' in source.metadata:
                            st.caption(f"📍 Chunk ID: {source.metadata['chunk_id']}")
    
    # Chat input box
    if prompt := st.chat_input("Ask a question about fluoride in drinking water..."):
        
        # Add user message to history
        st.session_state.messages.append({
            "role": "user",
            "content": prompt
        })
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate and display bot response
        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking (Gemini 2.5 Pro)..."):
                response, sources = st.session_state.chatbot.get_response(prompt)
                st.markdown(response)
                
                # Show sources
                if sources:
                    with st.expander("📚 View Sources"):
                        for i, source in enumerate(sources, 1):
                            st.markdown(f"**Source {i}:**")
                            # Display content preview
                            content_preview = source.page_content[:500]
                            if len(source.page_content) > 500:
                                content_preview += "..."
                            st.text(content_preview)
                            
                            # Display metadata
                            if 'chunk_id' in source.metadata:
                                st.caption(f"📍 Chunk ID: {source.metadata['chunk_id']}")
        
        # Add assistant message to history
        st.session_state.messages.append({
            "role": "assistant",
            "content": response,
            "sources": sources
        })


if __name__ == "__main__":
    main()
