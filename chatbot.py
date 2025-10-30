import streamlit as st
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import GoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import os

st.set_page_config(page_title="Fluoride Chatbot", page_icon="💧")

class RAGChatbot:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.vectorstore = Chroma(
            persist_directory="./chroma_db",
            embedding_function=self.embeddings
        )

    def build_qa_chain(self, google_api_key, model_name="gemini-pro"):
        template = """
        You are a helpful assistant answering correctly about fluoride in drinking water.
        Use only the provided context, don't guess. Cite context when needed.
        Context: {context}
        Question: {question}
        Answer:
        """
        PROMPT = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
        llm = GoogleGenerativeAI(
            model=model_name,
            google_api_key=google_api_key,
            temperature=0,
        )
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT},
        )

    def get_response(self, question):
        return self.qa_chain({"query": question})

def main():
    st.header("💧 Fluoride in Drinking Water Chatbot")
    google_api_key = st.sidebar.text_input("Enter Google Gemini API Key", type="password")
    model_name = st.sidebar.selectbox("Gemini Model", ["gemini-pro"])

    if not google_api_key:
        st.info("Please enter your Google Gemini API key in the sidebar.")
        st.stop()

    if "chatbot" not in st.session_state:
        st.session_state.chatbot = RAGChatbot()
        st.session_state.chatbot.build_qa_chain(google_api_key, model_name)

    question = st.text_input("Ask a question about fluoride in water:")

    if question:
        response = st.session_state.chatbot.get_response(question)
        st.write(response["result"])
        if "source_documents" in response and response["source_documents"]:
            st.write("Sources:")
            for doc in response["source_documents"]:
                st.text(doc.page_content[:350])

if __name__ == "__main__":
    main()



