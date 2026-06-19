import os
import fast_chunker  # The Rust module
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

# Configuration setup
DATA_PATH = "data/books"  # Put PDFs in a folder named data/books
DB_PATH = "vectorstore/db"  # Database


def run_ingestion():
    """Load PDF from the data folder"""

    # Ensure the directory exists
    if not os.path.exists(DATA_PATH):
        print(f"❌ Error: Please create the {DATA_PATH} folder and add your PDFs.")
        return

    # Load PDFs
    print("🚀 Loading documents")
    documents = []
    for file in os.listdir(DATA_PATH):
        if file.endswith(".pdf"):
            print(f"Loading: {file}")
            loader = PyPDFLoader(os.path.join(DATA_PATH, file))
            documents.extend(loader.load())

    # Rust Implementation Chunking Sequence
    print("🪓Spliting into chunks using Rust engine")
    chunks = []

    for doc in documents:
        raw_chunks = fast_chunker.chunk_text(
            doc.page_content, chunk_size=1000, chunk_overlap=150
        )

        # Take the raw string chunks from Rust and rebuild LaingChain documents
        for chunk_text in raw_chunks:
            chunks.append(
                Document(page_content=chunk_text, metadata=doc.metadata.copy())
            )

    print(f"✅ Created {len(chunks)} text chunks via Rust.")

    # Vectorising - turn the text into math
    print("Embedding and storing")

    # Use an AI model for embedding and storing of data
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Create and update the database, save to disk
    vector_db = Chroma.from_documents(
        # Feeds the small pieces of split text into the database
        # to be indexed and stored for searching.
        documents=chunks,
        # Tells the database which model to use for converting
        # text into searchable numerical values.
        embedding=embeddings,
        # Sets the folder where the database is saved to disk
        # so the data isn't lost when the script stops.
        persist_directory=DB_PATH,
    )

    # Use the vector_db variable to verify the collection count
    collection_count = vector_db._collection.count()
    print(f"📊 Database now contains {collection_count}")

    print(f"✅ Success! Knowledge base saved to {DB_PATH}")


if __name__ == "__main__":
    run_ingestion()
