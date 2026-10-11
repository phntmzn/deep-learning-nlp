import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE


# 1. Plot word/token embeddings using t-SNE
def tsne_plot(model, max_words=100):
    """
    Visualize embedding vectors in two dimensions.

    Args:
        model: Dictionary mapping token labels to embedding vectors.
        max_words: Maximum number of tokens to display.
    """
    labels = []
    tokens = []

    for word, vector in model.items():
        if len(labels) >= max_words:
            break

        labels.append(str(word))
        tokens.append(
            vector.detach().cpu().numpy()
            if isinstance(vector, torch.Tensor)
            else np.asarray(vector)
        )

    if len(tokens) < 2:
        print("At least two embedding vectors are required.")
        return

    tokens = np.asarray(tokens)

    # t-SNE perplexity must be less than the number of samples.
    perplexity = min(40, len(tokens) - 1)

    tsne_model = TSNE(
        perplexity=perplexity,
        n_components=2,
        init="pca",
        max_iter=1000,
        random_state=23,
    )

    new_values = tsne_model.fit_transform(tokens)

    plt.figure(figsize=(10, 10))

    for i, value in enumerate(new_values):
        x, y = value

        plt.scatter(x, y)
        plt.annotate(
            labels[i],
            xy=(x, y),
            xytext=(5, 2),
            textcoords="offset points",
            ha="right",
            va="bottom",
        )

    plt.title("Token Embeddings Visualized with t-SNE")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.tight_layout()
    plt.show()


# 2. Define the embedding model
VOCAB_SIZE = 100
EMBEDDING_DIM = 8
SEQUENCE_LENGTH = 10
BATCH_SIZE = 10

torch.manual_seed(42)
np.random.seed(42)


class EmbeddingModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=VOCAB_SIZE,
            embedding_dim=EMBEDDING_DIM,
        )

    def forward(self, x):
        return self.embedding(x)


model = EmbeddingModel()
model.eval()


# 3. Generate sample token IDs
input_array = np.random.randint(
    VOCAB_SIZE,
    size=(BATCH_SIZE, SEQUENCE_LENGTH),
)

input_tensor = torch.tensor(input_array, dtype=torch.long)


# 4. Generate embedding vectors
with torch.no_grad():
    output_tensor = model(input_tensor)

output_array = output_tensor.cpu().numpy()

print("Input shape:", input_array.shape)
print("Output shape:", output_array.shape)


# 5. Create a mapping from token ID to embedding vector
M = {}

for i in range(len(input_array)):
    for j in range(len(input_array[i])):
        token_id = int(input_array[i][j])

        # Keep the first embedding encountered for each token ID.
        if token_id not in M:
            M[token_id] = output_array[i][j]


# 6. Visualize the embeddings
tsne_plot(M)
