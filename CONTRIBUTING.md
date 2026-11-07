# 🤝 Contributing to RAG Library AI

Thank you for your interest in contributing.

RAG Library AI is an experimental Retrieval-Augmented Generation (RAG) application focused on grounded responses, semantic verification, and hallucination reduction. Contributions that improve reliability, performance, testing, documentation, and user experience are always welcome.

---

## ⚒️ Getting Started

### Clone the Repository

```bash
git clone https://github.com/reory/rag_library_ai.git
cd rag_library_ai
```

### Install Dependencies

```bash
uv sync
```

### Configure Environment Variables

Create a `.env` file in the project root:

```text
GOOGLE_API_KEY=your_api_key_here
```

### Ingest Documents

Build the vector database from your source documents:

```bash
python ingest.py
```

### Run the Application

```bash
streamlit run app.py
```

---

## 🧩 Contribution Guidelines

Please keep contributions:

- Small and focused
- Well documented
- Consistent with the existing code style
- Backed by tests where practical
- Aligned with the project's goals

### ✅ Good Contributions

- Bug fixes
- New tests
- Documentation improvements
- Performance enhancements
- User interface improvements
- Guardrail improvements
- Roadmap features

### ❌ Please Avoid

- Large unrelated changes in a single pull request
- Unnecessary dependencies
- Breaking existing functionality
- Major refactors without discussion

---

## 👨🏽‍💻 Development Notes

### Core Technologies

This project uses:

- Streamlit
- LangChain
- ChromaDB
- Google Gemini
- Sentence Transformers
- Rust (PyO3) via `fast_chunker`

### Rust Components

The `fast_chunker` package is used to accelerate document splitting during ingestion.

Contributors modifying the Rust implementation should ensure the package builds successfully before submitting a pull request.

---

## 🧪 Testing

Before submitting a pull request, please ensure:

```bash
python -m pytest
```

runs successfully.

Contributors are encouraged to add tests for:

- Retrieval pipelines
- Guardrail logic
- Document ingestion
- Semantic verification
- New roadmap features

---

## 📚 Test Data

When testing retrieval quality:

- Use relevant technical documentation.
- Verify source retrieval remains accurate.
- Check that guardrail verification behaves as expected.
- Confirm responses remain grounded in retrieved context.

---

## 📸 UI Changes

If your contribution modifies the user interface, please include screenshots in the pull request.

---

## 🛣️ Roadmap Areas

Contributions are especially welcome in:

- Source Citation
- Conversational Memory
- Semantic Input Routing
- Self-Correction Loops
- Multi-Document Knowledge Graphs
- Retrieval Quality Improvements
- Guardrail Enhancements

---

## 📜 License

By contributing, you agree that your contributions will be licensed under the MIT License.