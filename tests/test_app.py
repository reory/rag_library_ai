# Focus on format_docs and load_llm from app.py,
# mocking heavy dependencies so system does not hit real Chroma or Gemini.

import app


def test_format_docs_app(sample_docs):
    text = app.format_docs(sample_docs)
    assert "Python lists are mutable" in text
    assert "Tuples are immutable" in text


def test_load_llm_builds_chain(
    monkeypatch, fake_chroma_class, fake_embeddings, fake_llm_class
):
    # Monkeypatch heavy dependencies
    monkeypatch.setattr(
        app, "HuggingFaceEmbeddings", lambda model_name: fake_embeddings
    )
    monkeypatch.setattr(app, "Chroma", fake_chroma_class)
    monkeypatch.setattr(app, "ChatGoogleGenerativeAI", fake_llm_class)

    # Call the refactored initialization function
    retriever, prompt, llm, guardrail = app.load_rag_components()

    # Assert that all components were built correctly
    assert retriever is not None
    assert prompt is not None
    assert llm is not None
    assert guardrail is not None