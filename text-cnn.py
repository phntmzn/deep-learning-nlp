import sys
from collections import Counter

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


# ============================================================
# 1. CONFIGURATION
# ============================================================

MAX_WORDS = 1000
EMBEDDING_DIM = 100
BATCH_SIZE = 64
EPOCHS = 3
TEST_SIZE = 0.2
RANDOM_STATE = 36
LEARNING_RATE = 0.001

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)


# ============================================================
# 2. TOKENIZER
# ============================================================

class TextTokenizer:
    def __init__(self, max_words=1000):
        self.max_words = max_words
        self.word_index = {}

    def fit_on_texts(self, texts):
        word_counts = Counter()

        for text in texts:
            word_counts.update(str(text).lower().split())

        self.word_index = {
            word: index
            for index, (word, _) in enumerate(
                word_counts.most_common(self.max_words - 1),
                start=1
            )
        }

    def texts_to_sequences(self, texts):
        sequences = []

        for text in texts:
            sequence = [
                self.word_index[word]
                for word in str(text).lower().split()
                if word in self.word_index
            ]
            sequences.append(sequence)

        return sequences


def pad_sequences(sequences, max_length=None):
    if max_length is None:
        max_length = max(
            1,
            max((len(seq) for seq in sequences), default=0)
        )

    padded = torch.zeros(
        len(sequences),
        max_length,
        dtype=torch.long
    )

    for i, sequence in enumerate(sequences):
        # Match the default Keras pre-padding behavior.
        sequence = sequence[-max_length:]

        if sequence:
            padded[i, -len(sequence):] = torch.tensor(
                sequence,
                dtype=torch.long
            )

    return padded


# ============================================================
# 3. PYTORCH CNN MODEL
# ============================================================

class TextCNN(nn.Module):

    def __init__(
        self,
        vocab_size,
        embedding_dim,
        sequence_length,
        num_classes=2
    ):
        super().__init__()

        self.sequence_length = sequence_length

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim
        )

        self.conv1 = nn.Conv1d(
            in_channels=embedding_dim,
            out_channels=64,
            kernel_size=3,
            padding=1
        )

        self.conv2 = nn.Conv1d(
            in_channels=64,
            out_channels=32,
            kernel_size=3,
            padding=1
        )

        self.conv3 = nn.Conv1d(
            in_channels=32,
            out_channels=16,
            kernel_size=3,
            padding=1
        )

        self.flatten = nn.Flatten()

        self.dropout = nn.Dropout(p=0.2)

        self.output = nn.Linear(
            in_features=16 * sequence_length,
            out_features=num_classes
        )

    def forward(self, x):

        # Input: (batch, sequence_length)
        x = self.embedding(x)

        # Embedding output:
        # (batch, sequence_length, embedding_dim)

        # Conv1d expects:
        # (batch, channels, sequence_length)
        x = x.transpose(1, 2)

        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))

        x = self.flatten(x)
        x = self.dropout(x)

        # Return logits for CrossEntropyLoss.
        return self.output(x)


# ============================================================
# 4. KERAS-STYLE MODEL SUMMARY
# ============================================================

def model_summary(model):

    sequence_length = model.sequence_length
    embedding_dim = model.embedding.embedding_dim

    layers = [
        (
            "embedding_1 (Embedding)",
            f"(None, {sequence_length}, {embedding_dim})",
            model.embedding
        ),
        (
            "conv1d_1 (Conv1D)",
            f"(None, {sequence_length}, 64)",
            model.conv1
        ),
        (
            "conv1d_2 (Conv1D)",
            f"(None, {sequence_length}, 32)",
            model.conv2
        ),
        (
            "conv1d_3 (Conv1D)",
            f"(None, {sequence_length}, 16)",
            model.conv3
        ),
        (
            "flatten_1 (Flatten)",
            f"(None, {sequence_length * 16})",
            model.flatten
        ),
        (
            "dropout_1 (Dropout)",
            f"(None, {sequence_length * 16})",
            model.dropout
        ),
        (
            "dense_1 (Dense)",
            f"(None, {model.output.out_features})",
            model.output
        )
    ]

    print()
    print(
        f"{'Layer (type)':<29}"
        f"{'Output Shape':<26}"
        f"Param #"
    )

    print("=" * 67)

    for name, shape, layer in layers:

        params = sum(
            p.numel() for p in layer.parameters()
        )

        print(
            f"{name:<29}"
            f"{shape:<26}"
            f"{params:,}"
        )

    print("=" * 67)

    total_params = sum(
        p.numel() for p in model.parameters()
    )

    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    non_trainable_params = (
        total_params - trainable_params
    )

    print(f"Total params: {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")
    print(f"Non-trainable params: {non_trainable_params:,}")
    print()


# ============================================================
# 5. TRAINING FUNCTION
# ============================================================

def train_model(model, train_loader, criterion, optimizer):

    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in train_loader:

        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE)

        # Reset gradients.
        optimizer.zero_grad()

        # Forward pass.
        outputs = model(inputs)

        # Calculate loss.
        loss = criterion(outputs, labels)

        # Backpropagation.
        loss.backward()

        # Update weights.
        optimizer.step()

        batch_size = labels.size(0)

        total_loss += loss.item() * batch_size

        predictions = outputs.argmax(dim=1)

        correct += (
            predictions == labels
        ).sum().item()

        total += batch_size

    average_loss = total_loss / max(total, 1)
    accuracy = 100.0 * correct / max(total, 1)

    return average_loss, accuracy


# ============================================================
# 6. EVALUATION FUNCTION
# ============================================================

def evaluate_model(model, test_loader, criterion):

    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        for inputs, labels in test_loader:

            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(inputs)

            loss = criterion(outputs, labels)

            batch_size = labels.size(0)

            total_loss += loss.item() * batch_size

            predictions = outputs.argmax(dim=1)

            correct += (
                predictions == labels
            ).sum().item()

            total += batch_size

    average_loss = total_loss / max(total, 1)
    accuracy = 100.0 * correct / max(total, 1)

    return average_loss, accuracy


# ============================================================
# 7. MAIN PROGRAM
# ============================================================

def main():

    if len(sys.argv) < 2:
        print("Usage: python text_cnn.py <dataset.tsv>")
        sys.exit(1)

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    data = pd.read_csv(sys.argv[1], sep="\t")

    required_columns = {"text", "label"}

    if not required_columns.issubset(data.columns):
        raise ValueError(
            "Dataset must contain 'text' and 'label' columns."
        )

    data = data.dropna(
        subset=["text", "label"]
    ).copy()

    data["text"] = data["text"].astype(str)

    if len(data) < 5:
        raise ValueError(
            "Dataset is too small. Provide at least 5 examples."
        )

    label_names = sorted(data["label"].unique())

    if len(label_names) != 2:
        raise ValueError(
            "This script requires exactly two classes."
        )

    print(f"Using device: {DEVICE}")
    print(f"Dataset size: {len(data)}")

    # --------------------------------------------------------
    # Tokenize text
    # --------------------------------------------------------

    tokenizer = TextTokenizer(
        max_words=MAX_WORDS
    )

    tokenizer.fit_on_texts(data["text"].tolist())

    sequences = tokenizer.texts_to_sequences(
        data["text"].tolist()
    )

    X = pad_sequences(sequences)

    sequence_length = X.shape[1]

    print(f"Vocabulary size: {len(tokenizer.word_index)}")
    print(f"Sequence length: {sequence_length}")

    # --------------------------------------------------------
    # Encode labels
    # --------------------------------------------------------

    label_to_id = {
        label: index
        for index, label in enumerate(label_names)
    }

    y = torch.tensor(
        [
            label_to_id[label]
            for label in data["label"]
        ],
        dtype=torch.long
    )

    print(f"Labels: {label_to_id}")

    # --------------------------------------------------------
    # Split data
    # --------------------------------------------------------

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y
        )
    except ValueError as error:
        raise ValueError(
            "Could not split dataset. Provide enough examples "
            "in each class for a stratified train/test split."
        ) from error

    train_dataset = TensorDataset(
        X_train,
        y_train
    )

    test_dataset = TensorDataset(
        X_test,
        y_test
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    # --------------------------------------------------------
    # Build model
    # --------------------------------------------------------

    model = TextCNN(
        vocab_size=MAX_WORDS,
        embedding_dim=EMBEDDING_DIM,
        sequence_length=sequence_length,
        num_classes=2
    ).to(DEVICE)

    # Display Keras-style summary.
    model_summary(model)

    # --------------------------------------------------------
    # Configure loss and optimizer
    # --------------------------------------------------------

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    print("Training model...\n")

    for epoch in range(EPOCHS):

        loss, accuracy = train_model(
            model,
            train_loader,
            criterion,
            optimizer
        )

        print(
            f"Epoch {epoch + 1}/{EPOCHS} "
            f"- Loss: {loss:.4f} "
            f"- Accuracy: {accuracy:.2f}%"
        )

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    test_loss, test_accuracy = evaluate_model(
        model,
        test_loader,
        criterion
    )

    print("\nEvaluation results")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.2f}%")

    # --------------------------------------------------------
    # Save model and tokenizer
    # --------------------------------------------------------

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "vocabulary": tokenizer.word_index,
        "label_to_id": label_to_id,
        "sequence_length": sequence_length,
        "max_words": MAX_WORDS,
        "embedding_dim": EMBEDDING_DIM,
        "num_classes": 2
    }

    torch.save(
        checkpoint,
        "text_cnn.pth"
    )

    print("\nModel saved to text_cnn.pth")


if __name__ == "__main__":
    main()
