import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset


# ============================================================
# 1. CONFIGURATION
# ============================================================

DATA = ["character"]
TEST_DATA = ["character"]

SAMPLE_SIZE = 256
HIDDEN_SIZE = 100
EPOCHS = 10
BATCH_SIZE = 32
LEARNING_RATE = 0.001

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

print(f"Using device: {DEVICE}")


# ============================================================
# 2. CHARACTER ENCODING
# ============================================================

# Build a deterministic character vocabulary.
alphabet = sorted(
    set(c for word in DATA for c in word)
)

enc = LabelEncoder()
enc.fit(alphabet)

num_chars = len(alphabet)

char_to_id = {
    char: index
    for index, char in enumerate(enc.classes_)
}

id_to_char = {
    index: char
    for char, index in char_to_id.items()
}

print(f"Alphabet: {alphabet}")
print(f"Number of characters: {num_chars}")


def one_hot_encode(char):
    """Convert a character to a one-hot vector."""

    vector = np.zeros(num_chars, dtype=np.float32)
    vector[char_to_id[char]] = 1.0

    return vector


# ============================================================
# 3. PREPARE TRAINING SEQUENCES
# ============================================================

X_train = []
y_train = []

for word in DATA:

    for i in range(len(word) - 1):

        current_char = word[i]
        next_char = word[i + 1]

        X_train.append(
            one_hot_encode(current_char)
        )

        y_train.append(
            one_hot_encode(next_char)
        )


X_train = np.array(X_train, dtype=np.float32)
y_train = np.array(y_train, dtype=np.float32)

sample_len = len(X_train)

# Repeat the sequence to create training samples.
X_train = np.tile(
    X_train[np.newaxis, :, :],
    (SAMPLE_SIZE, 1, 1)
)

y_train = np.tile(
    y_train[np.newaxis, :, :],
    (SAMPLE_SIZE, 1, 1)
)

print(f"Training input shape: {X_train.shape}")
print(f"Training target shape: {y_train.shape}")


# ============================================================
# 4. PREPARE TEST SEQUENCES
# ============================================================

X_test = []
y_test = []

for word in TEST_DATA:

    for i in range(len(word) - 1):

        current_char = word[i]
        next_char = word[i + 1]

        X_test.append(
            one_hot_encode(current_char)
        )

        y_test.append(
            one_hot_encode(next_char)
        )


X_test = np.array(
    X_test,
    dtype=np.float32
).reshape(1, -1, num_chars)

y_test = np.array(
    y_test,
    dtype=np.float32
).reshape(1, -1, num_chars)

print(f"Test input shape: {X_test.shape}")
print(f"Test target shape: {y_test.shape}")


# ============================================================
# 5. DEFINE THE SIMPLE RNN
# ============================================================

class CharacterRNN(nn.Module):

    def __init__(
        self,
        input_size,
        hidden_size,
        output_size
    ):
        super().__init__()

        # Keras SimpleRNN equivalent.
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            nonlinearity="tanh"
        )

        # Keras TimeDistributed(Dense(...)) equivalent.
        # Applying Linear to a 3D tensor processes every time step.
        self.output_layer = nn.Linear(
            hidden_size,
            output_size
        )

    def forward(self, x):

        # x: (batch, sequence_length, input_size)
        rnn_output, hidden_state = self.rnn(x)

        # rnn_output:
        # (batch, sequence_length, hidden_size)

        logits = self.output_layer(rnn_output)

        # logits:
        # (batch, sequence_length, output_size)

        return logits


model = CharacterRNN(
    input_size=num_chars,
    hidden_size=HIDDEN_SIZE,
    output_size=num_chars
).to(DEVICE)


# ============================================================
# 6. DISPLAY MODEL SUMMARY
# ============================================================

def model_summary(model, sequence_length, input_size):

    hidden_size = model.rnn.hidden_size
    output_size = model.output_layer.out_features

    # Single-layer, unidirectional PyTorch RNN parameters:
    # input weights + hidden weights + two bias vectors.
    rnn_params = sum(
        p.numel()
        for p in model.rnn.parameters()
    )

    output_params = sum(
        p.numel()
        for p in model.output_layer.parameters()
    )

    layers = [
        (
            "input (OneHot)",
            f"(None, {sequence_length}, {input_size})",
            0
        ),
        (
            "simple_rnn (SimpleRNN)",
            f"(None, {sequence_length}, {hidden_size})",
            rnn_params
        ),
        (
            "time_distributed (Dense)",
            f"(None, {sequence_length}, {output_size})",
            output_params
        )
    ]

    print()
    print(
        f"{'Layer (type)':<30}"
        f"{'Output Shape':<30}"
        f"Param #"
    )
    print("=" * 75)

    for name, shape, params in layers:

        print(
            f"{name:<30}"
            f"{shape:<30}"
            f"{params:,}"
        )

    print("=" * 75)

    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(f"Total params: {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")
    print(f"Non-trainable params: {total_params - trainable_params:,}")
    print()


model_summary(
    model,
    sequence_length=sample_len,
    input_size=num_chars
)


# ============================================================
# 7. LOSS FUNCTION AND OPTIMIZER
# ============================================================

# One-hot targets with binary cross-entropy.
criterion = nn.BCEWithLogitsLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# 8. CREATE DATA LOADERS
# ============================================================

X_train_tensor = torch.tensor(
    X_train,
    dtype=torch.float32
)

y_train_tensor = torch.tensor(
    y_train,
    dtype=torch.float32
)

train_dataset = TensorDataset(
    X_train_tensor,
    y_train_tensor
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)


# ============================================================
# 9. TRAIN THE MODEL
# ============================================================

print("Training model...\n")

for epoch in range(EPOCHS):

    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in train_loader:

        inputs = inputs.to(DEVICE)
        targets = targets.to(DEVICE)

        # Clear previous gradients.
        optimizer.zero_grad()

        # Forward pass.
        logits = model(inputs)

        # Calculate loss.
        loss = criterion(logits, targets)

        # Backpropagation.
        loss.backward()

        # Update weights.
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)

        # Compare predicted characters with target characters.
        predictions = logits.argmax(dim=-1)
        expected = targets.argmax(dim=-1)

        correct += (
            predictions == expected
        ).sum().item()

        total += expected.numel()

    average_loss = total_loss / len(train_dataset)
    accuracy = 100.0 * correct / max(total, 1)

    print(
        f"Epoch {epoch + 1}/{EPOCHS} "
        f"- Loss: {average_loss:.4f} "
        f"- Accuracy: {accuracy:.2f}%"
    )


# ============================================================
# 10. PREDICT NEXT CHARACTERS
# ============================================================

model.eval()

X_test_tensor = torch.tensor(
    X_test,
    dtype=torch.float32
).to(DEVICE)

with torch.no_grad():

    logits = model(X_test_tensor)

    # Convert logits to probabilities.
    probabilities = torch.sigmoid(logits)

    # Select the most likely character at each time step.
    predictions = probabilities.argmax(dim=-1)

    predictions = predictions[0].cpu().numpy()


print("\nPredicted next characters:")

for char_id in predictions:
    print(id_to_char[int(char_id)], end="")

print()


# ============================================================
# 11. EVALUATE THE MODEL
# ============================================================

y_test_tensor = torch.tensor(
    y_test,
    dtype=torch.float32
).to(DEVICE)

with torch.no_grad():

    model.eval()

    logits = model(X_test_tensor)

    test_loss = criterion(
        logits,
        y_test_tensor
    ).item()

    predicted_ids = logits.argmax(dim=-1)

    expected_ids = y_test_tensor.argmax(dim=-1)

    test_accuracy = (
        predicted_ids == expected_ids
    ).float().mean().item() * 100

print("\nEvaluation results:")
print(f"Test loss: {test_loss:.4f}")
print(f"Test accuracy: {test_accuracy:.2f}%")


# ============================================================
# 12. SAVE THE MODEL
# ============================================================

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "alphabet": alphabet,
        "hidden_size": HIDDEN_SIZE,
        "num_chars": num_chars
    },
    "character_rnn.pth"
)

print("\nModel saved to character_rnn.pth")
