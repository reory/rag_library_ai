import os
import warnings

import streamlit as st
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings

from guardrail import RAGGuardrail

# Block Python's internal warnings if not fatal
warnings.filterwarnings("ignore", category=FutureWarning)

# Tell Transformers to only talk if it's a serious Error
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["STREAMLIT_LOG_LEVEL"] = "error"

# Load environment variables
load_dotenv()

# Page configuration - tab title and icon
st.set_page_config(page_title="Roy Peters", page_icon="😁", layout="centered")

# CUSTOM UI THEME + HEADER
st.markdown(
    """
    <style>
    .stApp {
        background-color: #e6f2ff;
        background-image: linear-gradient(to bottom right, #e6f2ff, #cce0ff);
        font-family: 'Segoe UI', sans-serif;
    }
    h1 {
        color: #1a1a1a;
        text-align: center;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    h3 {
        color: #004080;
        text-align: center;
        font-weight: 500;
        margin-top: 0rem;
    }
    .badge-container {
        text-align: center;
        margin-bottom: 25px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("<h1>RAG Library AI</h1>", unsafe_allow_html=True)
st.markdown("<h3>Ask anything based on Python.</h3>", unsafe_allow_html=True)

# Python library badges
st.markdown(
    """
    <div class="badge-container">
        <img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white">
        <img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white">
        <img src="https://img.shields.io/badge/Matplotlib-11557c?style=for-the-badge&logo=plotly&logoColor=white">
        <img src="https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white">
        <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white">
        <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white">
    </div>
    """,
    unsafe_allow_html=True
)

# --- HELPER FUNCTIONS ---
def clear_text_callback():
    """Reset the input field using a callback to avoid StreamlitAPIException"""
    st.session_state["query_input"] = ""

def format_docs(docs):
    """Converts Document objects into plain text for the LLM."""
    return "\n\n".join(doc.page_content for doc in docs)

@st.cache_resource
def load_rag_components():
    """Setup and cache individual RAG components for precise control."""
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_db = Chroma(persist_directory="vectorstore/db", embedding_function=embeddings)
    retriever = vector_db.as_retriever(search_kwargs={"k": 3})

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

    prompt = ChatPromptTemplate.from_template("""
    Answer based only on the provided context. If the answer isn't in the context, say so.

    Context: {context}
    Question: {question}
    """)

    # Instantiate our local guardrail checker (Threshold set to 60%)
    guardrail = RAGGuardrail(threshold=0.60)

    return retriever, prompt, llm, guardrail

# Unbundle components instead of using a collapsed chain linear pipeline
retriever, prompt_template, llm, guardrail = load_rag_components()

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
    with st.spinner("🔎 Searching the books and verfying facts..."):
        try:
            # Fetch relevant pages from the 3 books explicitly
            matched_docs = retriever.invoke(user_query)
            context_text = format_docs(matched_docs)

            if not context_text.strip():
                st.warning("🤖 The vector database could not find relevant content in the books.")
            
            else:
                # Format the custom prompt structure manually
                formatted_prompt = prompt_template.format(context=context_text, question=user_query)

                # Generate raw prediction from Gemini
                raw_response = llm.invoke(formatted_prompt).content

                # Push the response through the local Semantic Guardrail Filter
                is_safe, evaluation_score = guardrail.is_grounded(context_text, raw_response)

                # Route output determinstically based on factual similarity
                st.subheader("🤖 AI Response:")
                if is_safe:
                    st.write(raw_response)
                    st.caption(f"🛡️ Guardrail passed (Score: {evaluation_score:.2f})")
                
                else:
                    st.error(
                        "⚠️ Response Intercepted: "
                        "The AI attempted to hallucinate non-existent syntax or returned " 
                        "incoherent text patterns ungrounded by the books"
                    )
                    with st.expander("See details of the blocked response"):
                        st.write(f"**Semantic Alignment Score:** {evaluation_score:.2f} (Required: >= {guardrail.threshold:.2f})")
                        st.write(f"**Raw Intercepted Text:** {raw_response}")

        except Exception as e:
            st.error(f"An error occurred: {e}")