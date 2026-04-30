import os 
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma 
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_huggingface import HuggingFaceEmbeddings

# Configuration setup
DATA_PATH = "data/books" # Put PDFs in a folder named data/books
DB_PATH = "vectorstore/db" # Database

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

    # Chunking - Breaking books into 1000 character pieces
    print("Spliting into chunks")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150, # overlap helps maintain context for the AI 
        add_start_index=True # This keeps track of where the text was found
    )
    chunks = text_splitter.split_documents(documents)
    print(f"✅ Created {len(chunks)} text chunks.")

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
        persist_directory=DB_PATH
    )

    # Use the vector_db variable to verify the collection count
    collection_count = vector_db._collection.count()
    print(f"📊 Database now contains {collection_count}")

    print(f"✅ Success! Knowledge base saved to {DB_PATH}")

if __name__ == "__main__":
    run_ingestion()