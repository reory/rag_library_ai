from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings

from guardrail import RAGGuardrail

load_dotenv()

DB_PATH = "vectorstore/db"


def format_docs(docs):
    """Converts Document objects into plain text for the LLM."""
    return "\n\n".join(doc.page_content for doc in docs)


def ask_the_books(question):
    """
    Query the loaded books with the given questions,
    intercept hallucinations via guardrails, and return verified answers.
    """
    # Initialize RAG components individually
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_db = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    retriever = vector_db.as_retriever(search_kwargs={"k": 3})

    prompt_template = ChatPromptTemplate.from_template("""
    Answer based only on the provided context. If the answer isn't in the context, say so.

    Context: {context}
    Question: {question}
    """)

    # Instantiate our Guardrail Engine (matching the Streamlit threshold)
    guardrail = RAGGuardrail(threshold=0.60)

    print(f"\n🔎 Searching the books for: '{question}'...")
    
    try:
        # Fetch context chunks explicitly from the vector DB
        matched_docs = retriever.invoke(question)
        context_text = format_docs(matched_docs)

        if not context_text.strip():
            print("🤖 AI RESPONSE:\nThe vector database couldn't find relevant context in the books.")
            return

        # Format the raw text context into the prompt blueprint
        formatted_prompt = prompt_template.format(context=context_text, question=question)

        # Call Gemini to get the prediction text
        raw_response = llm.invoke(formatted_prompt).content

        # Audit the output using our semantic guardrail
        is_safe, evaluation_score = guardrail.is_grounded(context_text, raw_response)

        # Route terminal output based on factual score metrics
        print("\n🤖 AI RESPONSE:")
        if is_safe:
            print(raw_response)
            print(f"\n 🛡️  Guardrail status: PASSED (Factual Score: {evaluation_score:.2f})")
        else:
            print("⚠️  [BLOCKED] Response failed accuracy check! The model attempted to hallucinate non-existent information or generated garbage text.")
            print(f"👉 Score: {evaluation_score:.2f} (Required: >= 0.60)")
            print(f"👉 Raw Blocked Output: {raw_response}")

    except Exception as e:
        print(f"❌ An error occurred during processing: {e}")


if __name__ == "__main__":
    user_query = input("\n🤖 What would you like to ask your Python books? ")
    if user_query.strip():
        ask_the_books(user_query)