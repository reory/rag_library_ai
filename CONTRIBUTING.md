# Contributing to RAG Library AI

First off, thank you for considering contributing!

## How Can I Contribute?

### Reporting Bugs
* Check the Issues tab to see if the bug has already been reported.
* If not, open a new issue with a clear title and a description of how to reproduce the error.

### Suggesting Enhancements
* We are always looking to improve the RAG pipeline! 
* Suggestions for better chunking strategies or new LLM integrations are welcome.

### Pull Requests
1. **Fork** the repository.
2. **Create a branch** for your feature (`git checkout -b feature/AmazingFeature`).
3. **Set up your environment**:
   - Install dependencies: `pip install -r requirements.txt`
   - Set up your `.env` with a `GOOGLE_API_KEY`.
4. **Commit your changes** (`git commit -m 'Add some AmazingFeature'`).
5. **Push to the branch** (`git push origin feature/AmazingFeature`).
6. **Open a Pull Request**.

## Technical Stack
This project uses:
* **Streamlit** for the UI.
* **LangChain** for the RAG orchestration.
* **ChromaDB** for vector storage.
* **Gemini 2.5 Flash** as the primary LLM.

## Coding Standards
* Please follow PEP 8 guidelines for Python code.
* Ensure any new dependencies are added to `requirements.txt` or `pyproject.toml`

## License
By contributing, you agree that your contributions will be licensed under the project's MIT License.