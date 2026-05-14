from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

DB_PATH = "vectorstore/db"

# ✅ This converts Document objects into plain text for the LLM
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def ask_the_books(question):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_db = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    retriever = vector_db.as_retriever(search_kwargs={"k": 3})

    prompt = ChatPromptTemplate.from_template("""
    Answer based only on the provided context. If the answer isn't in the context, say so.

    Context: {context}
    Question: {question}
    """)

    qa_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    print(f"🔎 Searching for: {question}..")
    response = qa_chain.invoke(question)

    print("\n🤖 AI RESPONSE")
    print(response)

if __name__ == "__main__":
    user_query = input("\n🤖 What would you like to ask your Python books? ")
    ask_the_books(user_query)