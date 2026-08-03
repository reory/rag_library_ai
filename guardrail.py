from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class RAGGuardrail:
    def __init__(self, threshold=0.55):
        """
        Initialises a local embedding model to audit the RAG outputs.
        Using the same embedding model as the rest of the application.
        """

        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.threshold = threshold

        self.refusal_phrases = [
            "isn't in the context",
            "is not in the context",
            "not mentioned in the context",
            "does not provide",
            "does not mention",
            "does not contain",
            "i am sorry",
            "cannot answer"
        ]

    def is_grounded(self, context:str, ai_response: str) -> tuple[bool, float]:
        """
        Compares the generated answer against the text retrieved from the books.
        Returns (True, score) if safe, (False, score) if it's jumbled or a
        hallucination.
        """

        lowered_response = ai_response.lower()

        # If the AI is safely refusing, let it pass!
        for phrase in self.refusal_phrases:
            if phrase in lowered_response:
                print(" Guardrail Audit -> Safe Refusal Detected bypassing Math.")
                return True, 1.0

        # Encode both into a vector space
        embeddings = self.model.encode([context, ai_response])

        context_vec = embeddings[0].reshape(1, -1)
        response_vec = embeddings[1].reshape(1, -1)

        # Calculate cosine similarity score
        score = cosine_similarity(context_vec, response_vec)[0][0]

        # Returns evaluation pass status and the score metric
        return score >= self.threshold, score