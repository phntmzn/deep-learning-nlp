import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset


# ============================================================
# 1. CONFIGURATION
# ============================================================

np.random.seed(1234)
torch.manual_seed(1234)

DATA = [
    "xyzaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaxyz",
    "pqraaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaapqr"
]

TEST_DATA = [
    "xyzaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaxyz",
    "pqraaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaapqr"
]

SAMPLE_SIZE = 512
HIDDEN_SIZE = 100
BATCH_SIZE = 32
LEARNING_RATE = 0.001

# Prevent an endless training loop.
MAX_ITERATIONS = 100

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

print(f"Using device: {DEVICE}")


# ============================================================
# 2. CHARACTER ENCODING
# ============================================================

alphabet = sorted(
    set(char for word in DATA for char in word)
)

encoder = LabelEncoder()
encoder.fit(alphabet)

char_to_id = {
    char: index
    for index, char in enumerate(encoder.classes_)
}

id_to_char = {
    index: char
    for char, index in char_to_id.items()
}

NUM_CHARS = len(alphabet)

print(f"Alphabet: {alphabet}")
print(f"Number of characters: {NUM_CHARS}")


def one_hot_encode(char):
    """Convert a character into a one-hot vector."""

    vector = np.zeros(NUM_CHARS, dtype=np.float32)
    vector[char_to_id[char]] = 1.0

    return vector


def encode_sequences(words):
    """Create current-character and next-character pairs."""

    X = []
    y = []

    for word in words:
        for i in range(len(word) - 1):

            X.append(one_hot_encode(word[i]))
            y.append(one_hot_encode(word[i + 1]))

    return (
        np.asarray(X, dtype=np.float32),
        np.asarray(y, dtype=np.float32)
    )


# ============================================================
# 3. PREPARE TRAINING DATA
# ============================================================

X_train, y_train = encode_sequences(DATA)

sample_len = len(X_train)

# Repeat the transition sequences SAMPLE_SIZE times.
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
# 4. PREPARE TEST DATA
# ============================================================

X_test, y_test = encode_sequences(TEST_DATA)

test_len = len(X_test)

X_test = X_test[np.newaxis, :, :]
y_test = y_test[np.newaxis, :, :]

print(f"Test input shape: {X_test.shape}")
print(f"Test target shape: {y_test.shape}")


# ============================================================
# 5. PYTORCH LSTM MODEL
# ============================================================

class CharacterLSTM(nn.Module):

    def __init__(
        self,
        input_size,
        hidden_size,
        output_size
    ):
        super().__init__()

        # Equivalent to Keras LSTM(return_sequences=True).
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True
        )

        # Equivalent to TimeDistributed(Dense(...)).
        # nn.Linear operates on the last dimension of every
        # time step in the sequence.
        self.output_layer = nn.Linear(
            hidden_size,
            output_size
        )

    def forward(self, x):

        # x:
        # (batch_size, sequence_length, input_size)

        lstm_output, (hidden_state, cell_state) = self.lstm(x)

        # lstm_output:
        # (batch_size, sequence_length, hidden_size)

        logits = self.output_layer(lstm_output)

        # logits:
        # (batch_size, sequence_length, output_size)

        return logits


model = CharacterLSTM(
    input_size=NUM_CHARS,
    hidden_size=HIDDEN_SIZE,
    output_size=NUM_CHARS
).to(DEVICE)


# ============================================================
# 6. KERAS-STYLE MODEL SUMMARY
# ============================================================

def model_summary(model, sequence_length, num_chars):

    hidden_size = model.lstm.hidden_size

    layers = [
        (
            "input (OneHot)",
            f"(None, {sequence_length}, {num_chars})",
            0
        ),
        (
            "lstm_1 (LSTM)",
            f"(None, {sequence_length}, {hidden_size})",
            sum(
                p.numel()
                for p in model.lstm.parameters()
            )
        ),
        (
            "time_distributed (Dense)",
            f"(None, {sequence_length}, {num_chars})",
            sum(
                p.numel()
                for p in model.output_layer.parameters()
            )
        )
    ]

    print()
    print(
        f"{'Layer (type)':<30}"
        f"{'Output Shape':<32}"
        f"Param #"
    )
    print("=" * 77)

    for name, shape, params in layers:
        print(
            f"{name:<30}"
            f"{shape:<32}"
            f"{params:,}"
        )

    print("=" * 77)

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
    print(
        f"Non-trainable params: "
        f"{total_params - trainable_params:,}"
    )
    print()


model_summary(
    model,
    sequence_length=sample_len,
    num_chars=NUM_CHARS
)


# ============================================================
# 7. LOSS FUNCTION AND OPTIMIZER
# ============================================================

# The original uses binary_crossentropy with one-hot targets.
# BCEWithLogitsLoss combines sigmoid and binary cross-entropy.
criterion = nn.BCEWithLogitsLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# 8. TRAINING DATA LOADER
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
# 9. EVALUATION FUNCTION
# ============================================================

def evaluate(model, X, y):

    model.eval()

    X = torch.tensor(
        X,
        dtype=torch.float32
    ).to(DEVICE)

    y = torch.tensor(
        y,
        dtype=torch.float32
    ).to(DEVICE)

    with torch.no_grad():

        logits = model(X)

        loss = criterion(logits, y)

        predictions = logits.argmax(dim=-1)
        expected = y.argmax(dim=-1)

        accuracy = (
            predictions == expected
        ).float().mean().item()

    return loss.item(), accuracy


# ============================================================
# 10. ITERATIVE TRAINING
# ============================================================

print("Starting iterative training...\n")

for iteration in range(1, MAX_ITERATIONS + 1):

    # Evaluate before each training iteration, matching
    # the order of the original Keras example.
    test_loss, test_accuracy = evaluate(
        model,
        X_test,
        y_test
    )

    print(
        f"[Iteration {iteration}] "
        f"loss={test_loss:.4f}, "
        f"accuracy={test_accuracy:.4f}"
    )

    # Stop if every next-character prediction is correct.
    if test_accuracy >= 1.0:
        print("Target accuracy reached.")
        break

    model.train()

    total_loss = 0.0

    for inputs, targets in train_loader:

        inputs = inputs.to(DEVICE)
        targets = targets.to(DEVICE)

        optimizer.zero_grad()

        logits = model(inputs)

        loss = criterion(logits, targets)

        loss.backward()

        optimizer.step()

        total_loss += loss.item() * inputs.size(0)

    average_loss = total_loss / len(train_dataset)

    print(f"  Training loss: {average_loss:.4f}")

else:
    print("Maximum training iterations reached.")


# ============================================================
# 11. PREDICT CHARACTERS
# ============================================================

model.eval()

X_test_tensor = torch.tensor(
    X_test,
    dtype=torch.float32
).to(DEVICE)

with torch.no_grad():

    logits = model(X_test_tensor)

    # Select the highest-scoring character at each time step.
    predictions = logits.argmax(dim=-1)[0].cpu().numpy()


print("\nPredicted next-character sequence:")

predicted_text = "".join(
    id_to_char[int(char_id)]
    for char_id in predictions
)

print(predicted_text)


# ============================================================
# 12. FINAL EVALUATION
# ============================================================

final_loss, final_accuracy = evaluate(
    model,
    X_test,
    y_test
)

print("\nFinal evaluation:")
print(f"Test loss: {final_loss:.4f}")
print(f"Test accuracy: {final_accuracy:.4f}")


# ============================================================
# 13. SAVE MODEL
# ============================================================

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "alphabet": alphabet,
        "hidden_size": HIDDEN_SIZE,
        "num_chars": NUM_CHARS
    },
    "character_lstm.pth"
)

print("\nModel saved to character_lstm.pth")
