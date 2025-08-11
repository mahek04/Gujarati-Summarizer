# summarizers/extractive.py
"""
Extractive Summarizer
- Uses Sentence-BERT embeddings to rank sentences by semantic centrality.
- Returns top N sentences as summary.
- GPU-ready (will use CUDA if available).
"""

import nltk
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List

# Download NLTK data (first time)
nltk.download("punkt", quiet=True)

class ExtractiveSummarizer:
    def __init__(self, model_name: str = "paraphrase-MiniLM-L6-v2"):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, device=device)

    def summarize(self, text: str, num_sentences: int = 3) -> str:
        """
        Summarize the text by selecting the most central sentences.
        """
        if not text.strip():
            return ""

        # Split into sentences
        sentences = nltk.sent_tokenize(text)
        if len(sentences) <= num_sentences:
            return text  # Already short enough

        # Encode sentences
        embeddings = self.model.encode(sentences, convert_to_tensor=True)

        # Compute centrality scores (average similarity to all others)
        cosine_matrix = cosine_similarity(embeddings.cpu(), embeddings.cpu())
        scores = cosine_matrix.mean(axis=1)

        # Pick top N sentence indices
        ranked_indices = scores.argsort()[-num_sentences:][::-1]
        top_sentences = [sentences[i] for i in sorted(ranked_indices)]

        return " ".join(top_sentences)
