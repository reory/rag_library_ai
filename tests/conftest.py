"""
Shared fixtures: fake docs, fake embeddings, fake Chroma,
fake LLM, and a helper to monkeypatch environment
"""

import os
import types

import pytest


@pytest.fixture(autouse=True)
def patch_heavy_imports_globally(monkeypatch, fake_chroma_class, fake_llm_class, fake_embeddings):
    """
    Globally intercepts and replaces real LangChain/Chroma/Google classes 
    with light fake equivalents before any 
    module initialization can trigger network/disk I/O.
    """

    # Patch Chroma vectorstore
    monkeypatch.setattr("langchain_chroma.Chroma", fake_chroma_class, raising=False)
    monkeypatch.setattr(
        "langchain_community.vectorstores.Chroma", 
        fake_chroma_class, 
        raising=False
    )
    
    # Patch Google / HuggingFace Embeddings
    monkeypatch.setattr(
        "langchain_google_genai.GoogleGenerativeAIEmbeddings", 
        lambda *a, **k: fake_embeddings, 
        raising=False
    )
    monkeypatch.setattr(
        "langchain_community.embeddings.HuggingFaceEmbeddings", 
        lambda *a, **k: 
        fake_embeddings, 
        raising=False
    )
    
    # Patch LLM / Chat models
    monkeypatch.setattr(
        "langchain_google_genai.ChatGoogleGenerativeAI", 
        fake_llm_class, 
        raising=False
    )


@pytest.fixture(autouse=True)
def fake_str_output_parser(monkeypatch):
    class FakeParser:
        def __call__(self, input):
            return self.invoke(input)

        def __ror__(self, other):
            return self

        def __or__(self, other):
            return self

        def invoke(self, input):
            return input

    monkeypatch.setattr(
        "langchain_core.output_parsers.StrOutputParser", lambda: FakeParser()
    )


@pytest.fixture(autouse=True)
def set_test_env(monkeypatch, tmp_path):
    """Ensure a clean env and temp vectorstore path for tests."""

    monkeypatch.setenv("GOOGLE_API_KEY", "DUMMY_KEY")
    monkeypatch.chdir(tmp_path)
    yield


@pytest.fixture
def sample_docs():
    """Simple mock 'Document' objects with page content and metadata."""

    Doc = types.SimpleNamespace
    return [
        Doc(page_content="Python lists are mutable sequences.", metadata={"page": 10}),
        Doc(
            page_content="Tuples are immutable sequences in Python.",
            metadata={"page": 11},
        ),
    ]


@pytest.fixture
def fake_embeddings():
    """Fake embedding model that returns deterministic vectors."""

    class FakeEmbeddings:
        def embed_documents(self, texts):
            return [[float(len(t))] for t in texts]

        def embed_query(self, text):
            return [float(len(text))]

    return FakeEmbeddings()


@pytest.fixture
def fake_chroma_class(sample_docs):
    """Fake chroma class to stand in for Chroma DB."""

    class FakeRetriever:
        def __init__(self, docs):
            self._docs = docs

        def get_relevant_documents(self, query):
            return self._docs

        # Added modern invoke method to support newer LangChain versions
        def invoke(self, query, *args, **kwargs):
            return self.get_relevant_documents(query)

        def __call__(self, query):
            return self.get_relevant_documents(query)

        def __or__(self, other):
            def runnable(_):
                return {"context": other(self._docs)}

            return runnable

        def __ror__(self, other):
            return self

    class FakeCollection:
        def __init__(self, docs):
            self._docs = docs

        def count(self):
            return len(self._docs)

    class FakeChroma:
        def __init__(self, persist_directory=None, embedding_function=None, **kwargs):
            self._docs = sample_docs
            self._collection = FakeCollection(self._docs)

            if persist_directory:
                os.makedirs(persist_directory, exist_ok=True)

        @classmethod
        def from_documents(cls, documents, embedding, persist_directory=None):
            inst = cls(
                persist_directory=persist_directory, embedding_function=embedding
            )
            inst._docs = documents
            inst._collection = FakeCollection(documents)
            return inst

        def as_retriever(self, search_kwargs=None):
            return FakeRetriever(self._docs)

    return FakeChroma


@pytest.fixture
def fake_llm_class():
    """Fake llm that echoes the question and context length."""

    class FakeLLM:
        def __init__(self, model=None, temperature=0, **kwargs):
            pass

        def __call__(self, input):
            return self.invoke(input)

        def __ror__(self, other):
            return self

        def __or__(self, other):
            return self

        def invoke(self, input):
            # Returns an object with a .content property to match modern Chat model outputs
            return types.SimpleNamespace(content="FAKE_LLM_RESPONSE")

    return FakeLLM
