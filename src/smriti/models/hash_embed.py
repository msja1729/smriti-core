"""Dev-only embedder for M3: bag of words hashed into `dim` buckets. No model, no network.
Replaced by the hosted NVIDIA embedding model in M4; benchmark numbers never use this."""
import hashlib
import re


class HashEmbedder:
    def __init__(self, dim: int = 256):
        self.dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        for text in texts:
            vec = [0.0] * self.dim
            for word in re.findall(r"\w+", text.lower()):
                vec[int(hashlib.sha256(word.encode()).hexdigest(), 16) % self.dim] += 1.0
            vec[0] += 1e-3                  # never the zero vector
            out.append(vec)
        return out
