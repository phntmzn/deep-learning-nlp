# embedding_classifier.py

import re
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# --------------------------------------------------
# Configuration
# --------------------------------------------------

MAX_LEN = 100
EMBEDDING_DIM = 100
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.001

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"

torch.manual_seed(42)


# --------------------------------------------------
# Save learned embeddings
# --------------------------------------------------

def save_embedding(output_file, weights, vocabulary):
    """
    Save embeddings in text format.

    First line:
        vocabulary_size embedding_dimension

    Subsequent lines:
        token embedding_value_1 embedding_value_2 ...
    """
    reverse_vocab = {index: word for word, index in vocabulary.items()}

    weights = weights.detach().cpu()

    with open(output_file, "w", encoding="utf-8") as file:
        file.write(f"{len(vocabulary)} {weights.shape[1]}\n")

        for index in sorted(reverse_vocab):
            word = reverse_vocab[index]
            vector = weights[index].tolist()

            file.write(
                word + " " + " ".join(f"{value:.6f}" for value in vector) + "\n"
            )

    print(f"Embeddings saved to: {output_file}")


# --------------------------------------------------
# Read input data
# --------------------------------------------------

def get_lines(file_path):
    """Read non-empty lines from a UTF-8 text file."""
    with open(file_path, "r", encoding="utf-8") as file:
        return [line.rstrip() for line in file if line.strip()]


# --------------------------------------------------
# Text preprocessing
# --------------------------------------------------

def normalize_word(word):
    """Lowercase a word and remove selected punctuation."""
    return re.sub(r"""[.,:;'"!?()]+""", "", word.lower())


def tokenize(sentence):
    """Split a sentence into normalized tokens."""
    return [
        normalize_word(word)
        for word in sentence.split()
        if normalize_word(word)
    ]


def create_vocabulary(vocabulary, sentences):
    """
    Add tokens to the vocabulary without changing existing IDs.

    Index 0 is reserved for padding.
    Index 1 is reserved for unknown words.
    """
    for sentence in sentences:
        text, _ = parse_labeled_line(sentence)

        for word in tokenize(text):
            if word not in vocabulary:
                vocabulary[word] = len(vocabulary)


def parse_labeled_line(line):
    """
    Parse a line formatted as:

        sentence<TAB>label

    Returns:
        (sentence, integer_label)

    Raises:
        ValueError if the line is malformed.
    """
    match = re.match(r"^([^\t]+)\t([01])$", line.strip())

    if not match:
        raise ValueError(
            f"Invalid data line: {line!r}. "
            "Expected sentence<TAB>0 or sentence<TAB>1."
        )

    sentence = match.group(1)
    label = int(match.group(2))

    return sentence, label


def encode_sentence(sentence, vocabulary, max_len):
    """Convert a sentence to token IDs and pad or truncate it."""
    words = tokenize(sentence)

    token_ids = [
        vocabulary.get(word, vocabulary[UNK_TOKEN])
        for word in words
    ]

    # Truncate sequences longer than max_len.
    token_ids = token_ids[:max_len]

    # Post-padding: append zeros until the sequence reaches max_len.
    token_ids.extend([vocabulary[PAD_TOKEN]] * (max_len - len(token_ids)))

    return token_ids


# --------------------------------------------------
# Process training data
# --------------------------------------------------

def process_training_data(text_file, max_len):
    """
    Read labeled training data and construct the vocabulary.

    Expected format per line:
        sentence<TAB>label
    """
    sentences = get_lines(text_file)

    vocabulary = {
        PAD_TOKEN: 0,
        UNK_TOKEN: 1,
    }

    parsed_data = [parse_labeled_line(line) for line in sentences]

    # Build vocabulary using training text only.
    create_vocabulary(vocabulary, sentences)

    data = []
    labels = []

    for sentence, label in parsed_data:
        data.append(encode_sentence(sentence, vocabulary, max_len))
        labels.append(label)

    return data, labels, vocabulary


# --------------------------------------------------
# Process test data
# --------------------------------------------------

def process_test_data(text_file, vocabulary, max_len):
    """
    Process test data using the training vocabulary.

    Unknown test words map to <unk>.
    The vocabulary is not expanded with test data.
    """
    sentences = get_lines(text_file)

    data = []
    labels = []

    for line in sentences:
        sentence, label = parse_labeled_line(line)

        data.append(encode_sentence(sentence, vocabulary, max_len))
        labels.append(label)

    return data, labels


# --------------------------------------------------
# PyTorch model
# --------------------------------------------------

class EmbeddingClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, max_len):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=0,
        )

        self.flatten = nn.Flatten()

        self.classifier = nn.Linear(
            embedding_dim * max_len,
            1,
        )

    def forward(self, x):
        x = self.embedding(x)
        x = self.flatten(x)
        x = self.classifier(x)

        # Return logits. BCEWithLogitsLoss applies sigmoid internally.
        return x.squeeze(-1)


# --------------------------------------------------
# Training
# --------------------------------------------------

def train_model(model, data, labels, epochs, batch_size, learning_rate):
    dataset = TensorDataset(
        torch.tensor(data, dtype=torch.long),
        torch.tensor(labels, dtype=torch.float32),
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
    )

    for epoch in range(epochs):
        model.train()

        total_loss = 0.0
        correct = 0
        total = 0

        for inputs, targets in loader:
            optimizer.zero_grad()

            logits = model(inputs)
            loss = criterion(logits, targets)

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * inputs.size(0)

            predictions = (logits >= 0).float()
            correct += (predictions == targets).sum().item()
            total += targets.size(0)

        average_loss = total_loss / total
        accuracy = correct / total

        print(
            f"Epoch [{epoch + 1:03d}/{epochs}] "
            f"Loss: {average_loss:.4f} "
            f"Accuracy: {accuracy:.4f}"
        )


# --------------------------------------------------
# Evaluation
# --------------------------------------------------

def evaluate_model(model, data, labels, batch_size=32):
    dataset = TensorDataset(
        torch.tensor(data, dtype=torch.long),
        torch.tensor(labels, dtype=torch.float32),
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    criterion = nn.BCEWithLogitsLoss()

    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in loader:
            logits = model(inputs)
            loss = criterion(logits, targets)

            total_loss += loss.item() * inputs.size(0)

            predictions = (logits >= 0).float()
            correct += (predictions == targets).sum().item()
            total += targets.size(0)

    if total == 0:
        raise ValueError("The test dataset is empty.")

    return total_loss / total, correct / total


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():
    if len(sys.argv) != 3:
        print(
            "Usage: python embedding_classifier.py "
            "<train_file.txt> <test_file.txt>"
        )
        sys.exit(1)

    train_file = Path(sys.argv[1])
    test_file = Path(sys.argv[2])

    if not train_file.is_file():
        sys.exit(f"Training file not found: {train_file}")

    if not test_file.is_file():
        sys.exit(f"Test file not found: {test_file}")

    # Load and preprocess datasets.
    data, labels, vocabulary = process_training_data(
        train_file,
        MAX_LEN,
    )

    test_data, test_labels = process_test_data(
        test_file,
        vocabulary,
        MAX_LEN,
    )

    if not data:
        sys.exit("The training dataset is empty.")

    if not test_data:
        sys.exit("The test dataset is empty.")

    print(f"Training examples: {len(data)}")
    print(f"Test examples: {len(test_data)}")
    print(f"Vocabulary size: {len(vocabulary)}")

    # Create the model.
    model = EmbeddingClassifier(
        vocab_size=len(vocabulary),
        embedding_dim=EMBEDDING_DIM,
        max_len=MAX_LEN,
    )

    # Train the model.
    train_model(
        model,
        data,
        labels,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
    )

    # Evaluate the model.
    loss, accuracy = evaluate_model(
        model,
        test_data,
        test_labels,
        batch_size=BATCH_SIZE,
    )

    print(f"\nTest loss: {loss:.4f}")
    print(f"Test accuracy: {accuracy:.4f}")

    # Export the learned embedding table.
    save_embedding(
        "embedding_labeled.txt",
        model.embedding.weight,
        vocabulary,
    )

    # Save the model and vocabulary for later use.
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "vocabulary": vocabulary,
            "max_len": MAX_LEN,
            "embedding_dim": EMBEDDING_DIM,
        },
        "embedding_classifier.pt",
    )

    print("Model saved to: embedding_classifier.pt")


if __name__ == "__main__":
    main()
