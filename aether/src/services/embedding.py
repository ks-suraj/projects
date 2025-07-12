from sentence_transformers import SentenceTransformer
from typing import List, Union
import numpy as np

class EmbeddingEngine:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        print(f"✅ Loaded embedding model: {model_name}")

    def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        elif not texts:
            return np.array([])
        return self.model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
