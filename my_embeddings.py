from crewai_tools.utils.embeddings import BaseEmbeddingFunction
import requests

class OllamaEmbedding(BaseEmbeddingFunction):
    def __init__(self, model="gemma:2b", url="http://localhost:11434/api/embeddings"):
        self.model = model
        self.url = url

    def __call__(self, texts):
        # expects a list of strings
        if isinstance(texts, str):
            texts = [texts]
        embeddings = []
        for t in texts:
            response = requests.post(
                self.url,
                json={"model": self.model, "text": t}
            )
            embeddings.append(response.json()["embedding"])
        return embeddings
