import streamlit as st
import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import warnings

# Block Python's internal warnings if not fatal
warnings.filterwarnings("ignore", category=FutureWarning)

# Tell Transformers to only talk if it's a serious Error
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["STREAMLIT_LOG_LEVEL"] = "error"

# Load File
load_dotenv()

# Page configuration - tab title and icon
st.set_page_config(page_title="Roy Peters", page_icon="😁")
st.title("RAG Library AI")
st.markdown("Ask anything based on your uploaded PDFs.")

# --- HELPER FUNCTIONS ---

# 
def clear_text_callback():
    """Reset the input field using a callback to avoid StreamlitAPIException"""
    st.session_state["query_input"] = ""

def format_docs(docs):
    """Converts Document objects into plain text for the LLM."""
    return "\n\n".join(doc.page_content for doc in docs)

# 
@st.cache_resource
def load_llm():
    """Setup the AI Brain."""
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_db = Chroma(persist_directory="vectorstore/db", embedding_function=embeddings)
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    retriever = vector_db.as_retriever(search_kwargs={"k": 3})

    prompt = ChatPromptTemplate.from_template("""
    Answer based only on the provided context. If the answer isn't in the context, say so.

    Context: {context}
    Question: {question}
    """)

    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

qa_chain = load_llm()

# --- CHAT INTERFACE ---

col1, col2 = st.columns([5, 1])

with col1:
    # Adding a key allows us to clear the input via session_state
    user_query = st.text_input("💬 Please enter your question:", key="query_input")

with col2:
    st.write("##") 
    # Use on_click callback. This runs BEFORE the next script execution,
    # allowing us to modify the state safely.
    st.button("Clear 🗑️", on_click=clear_text_callback)

# Processing the query
if user_query:
    with st.spinner("🔎 Searching the books..."):
        try:
            response = qa_chain.invoke(user_query)
            st.subheader("🤖 AI Response:")
            st.write(response)
        except Exception as e:
            st.error(f"An error occurred: {e}")