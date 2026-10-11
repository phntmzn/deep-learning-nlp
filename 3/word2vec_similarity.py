# word2vec_similarity.py

import os
import sys

import torch
import torch.nn.functional as F
from gensim.models import KeyedVectors


def main():
    if len(sys.argv) != 2:
        print(
            "Usage: python word2vec_similarity.py "
            "<word2vec_vectors.txt>"
        )
        sys.exit(1)

    vector_file = sys.argv[1]

    if not os.path.isfile(vector_file):
        sys.exit(f"Embedding file not found: {vector_file}")

    # 1. Load pretrained Word2Vec vectors from a text file.
    # Expected format:
    # vocabulary_size embedding_dimension
    # word1 value1 value2 ...
    w2v = KeyedVectors.load_word2vec_format(
        vector_file,
        binary=False,
        unicode_errors="ignore",
    )

    print(f"Vocabulary size: {len(w2v.key_to_index)}")
    print(f"Embedding dimension: {w2v.vector_size}")

    # 2. Convert all vectors into a PyTorch tensor.
    words = list(w2v.index_to_key)

    vectors = torch.tensor(
        w2v.vectors.copy(),
        dtype=torch.float32,
    )

    # Normalize vectors so dot products equal cosine similarities.
    normalized_vectors = F.normalize(vectors, p=2, dim=1)

    # 3. Find the three most similar words for each vocabulary word.
    for index, word in enumerate(words):
        similarities = torch.mv(
            normalized_vectors,
            normalized_vectors[index],
        )

        # Exclude the word itself.
        similarities[index] = -float("inf")

        top_k = min(3, len(words) - 1)

        if top_k <= 0:
            print(word, [])
            continue

        scores, indices = torch.topk(similarities, k=top_k)

        similar_words = [
            (words[i], round(score.item(), 4))
            for i, score in zip(indices.tolist(), scores)
        ]

        print(word, similar_words)


if __name__ == "__main__":
    main()
