import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# ============================================================
# 1. CONFIGURATION
# ============================================================

VOCAB_SIZE = 100
EMBEDDING_DIM = 8
SEQUENCE_LENGTH = 10
BATCH_SIZE = 10

torch.manual_seed(42)
np.random.seed(42)


# ============================================================
# 2. DEFINE THE EMBEDDING MODEL
# ============================================================

class EmbeddingModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=VOCAB_SIZE,
            embedding_dim=EMBEDDING_DIM
        )

    def forward(self, x):
        return self.embedding(x)


model = EmbeddingModel()


# ============================================================
# 3. CREATE INPUT DATA
# ============================================================

# Generate 10 sequences, each containing 10 token IDs.
input_array = np.random.randint(
    VOCAB_SIZE,
    size=(BATCH_SIZE, SEQUENCE_LENGTH)
)

input_tensor = torch.tensor(
    input_array,
    dtype=torch.long
)

print("Input array:")
print(input_array)

print("\nInput shape:")
print(input_tensor.shape)


# ============================================================
# 4. CONFIGURE THE OPTIMIZER AND LOSS
# ============================================================

# Keras: model.compile('rmsprop', 'mse')
optimizer = optim.RMSprop(
    model.parameters(),
    lr=0.001
)

criterion = nn.MSELoss()


# ============================================================
# 5. GENERATE EMBEDDINGS
# ============================================================

model.eval()

with torch.no_grad():
    output_tensor = model(input_tensor)

output_array = output_tensor.numpy()


# ============================================================
# 6. DISPLAY OUTPUT
# ============================================================

print("\nEmbedding output:")
print(output_array)

print("\nOutput shape:")
print(output_array.shape)

print("\nModel parameters:")
print(
    sum(
        parameter.numel()
        for parameter in model.parameters()
    )
)
