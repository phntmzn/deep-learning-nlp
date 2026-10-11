# word2vec_skipgram.py

import random
import re
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# --------------------------------------------------
# Configuration
# --------------------------------------------------

WINDOW_SIZE = 3
VECTOR_DIM = 100
BATCH_SIZE = 100
STEPS_PER_EPOCH = 100
LEARNING_RATE = 0.001

torch.manual_seed(42)
random.seed(42)


# --------------------------------------------------
# Text preprocessing
# --------------------------------------------------

def getLines(file_path):
    """Read sentences from a text file."""
    with open(file_path, "r", encoding="utf-8") as file:
        return [line.strip() for line in file if line.strip()]


def normalize_word(word):
    """Lowercase a word and remove selected punctuation."""
    return re.sub(r"""[.,:;'"!?()]+""", "", word.lower())


def create_vocabulary(vocabulary, sentences):
    """Create a word-to-index vocabulary."""
    vocabulary["<unk>"] = 0

    for sentence in sentences:
        for word in sentence.split():
            word = normalize_word(word)

            if word and word not in vocabulary:
                vocabulary[word] = len(vocabulary)


# --------------------------------------------------
# Skip-gram pairs and negative sampling
# --------------------------------------------------

def process_data(textFile, window_size):
    """
    Create positive and negative target-context pairs.

    Positive pair:
        Two words appearing within the context window.

    Negative pair:
        A target word paired with a randomly sampled word
        that is not one of its actual context words.
    """
    sentences = getLines(textFile)

    vocabulary = {}
    create_vocabulary(vocabulary, sentences)

    vocabulary_size = len(vocabulary)
    couples = []
    labels = []

    if vocabulary_size < 2:
        raise ValueError(
            "At least two vocabulary entries are required."
        )

    for sentence in sentences:
        words = [
            vocabulary[normalize_word(word)]
            for word in sentence.split()
            if normalize_word(word)
        ]

        # Skip sentences with fewer than two words.
        if len(words) < 2:
            continue

        for i, target_word in enumerate(words):
            start = max(0, i - window_size)
            end = min(len(words), i + window_size + 1)

            positive_contexts = [
                words[j]
                for j in range(start, end)
                if j != i
            ]

            for context_word in positive_contexts:
                couples.append((target_word, context_word))
                labels.append(1)

                # Sample a negative word that is not a true
                # context word for this target occurrence.
                forbidden = set(positive_contexts)
                forbidden.add(target_word)

                candidates = [
                    idx
                    for idx in range(1, vocabulary_size)
                    if idx not in forbidden
                ]

                if candidates:
                    negative_word = random.choice(candidates)
                    couples.append((target_word, negative_word))
                    labels.append(0)

    if not couples:
        raise ValueError(
            "No training pairs were generated. "
            "Use a larger text corpus."
        )

    return vocabulary, couples, labels


# --------------------------------------------------
# PyTorch Skip-gram model
# --------------------------------------------------

class SkipGramModel(nn.Module):
    def __init__(self, vocabulary_size, vector_dim):
        super().__init__()

        self.target_embedding = nn.Embedding(
            vocabulary_size,
            vector_dim,
        )

        self.context_embedding = nn.Embedding(
            vocabulary_size,
            vector_dim,
        )

        # Initialize embeddings with small random values.
        nn.init.uniform_(
            self.target_embedding.weight,
            -0.5 / vector_dim,
            0.5 / vector_dim,
        )

        nn.init.zeros_(self.context_embedding.weight)

    def forward(self, target_words, context_words):
        """
        Compute a compatibility score for each word pair.

        Inputs:
            target_words:  (batch_size,)
            context_words: (batch_size,)

        Output:
            logits:        (batch_size,)
        """
        target_vectors = self.target_embedding(target_words)
        context_vectors = self.context_embedding(context_words)

        # Dot product between each target and context vector.
        logits = (target_vectors * context_vectors).sum(dim=1)

        return logits


# --------------------------------------------------
# Save embeddings
# --------------------------------------------------

def save_embedding(output_file, weights, vocabulary):
    """Save embeddings in Word2Vec text format."""
    reverse_vocabulary = {
        index: word
        for word, index in vocabulary.items()
    }

    weights = weights.detach().cpu()

    with open(output_file, "w", encoding="utf-8") as file:
        file.write(
            f"{len(vocabulary)} {weights.shape[1]}\n"
        )

        for index in sorted(reverse_vocabulary):
            word = reverse_vocabulary[index]
            vector = weights[index].tolist()

            file.write(
                word + " "
                + " ".join(f"{value:.6f}" for value in vector)
                + "\n"
            )

    print(f"Embeddings saved to {output_file}")


# --------------------------------------------------
# Training
# --------------------------------------------------

def train_model(model, couples, labels, epochs, device):
    target_words = torch.tensor(
        [pair[0] for pair in couples],
        dtype=torch.long,
    )

    context_words = torch.tensor(
        [pair[1] for pair in couples],
        dtype=torch.long,
    )

    pair_labels = torch.tensor(
        labels,
        dtype=torch.float32,
    )

    dataset = TensorDataset(
        target_words,
        context_words,
        pair_labels,
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    model.to(device)

    for epoch in range(epochs):
        model.train()

        total_loss = 0.0
        total_correct = 0
        total_examples = 0

        for step, (targets, contexts, batch_labels) in enumerate(loader):
            if step >= STEPS_PER_EPOCH:
                break

            targets = targets.to(device)
            contexts = contexts.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()

            logits = model(targets, contexts)
            loss = criterion(logits, batch_labels)

            loss.backward()
            optimizer.step()

            batch_size = batch_labels.size(0)

            total_loss += loss.item() * batch_size

            predictions = (logits >= 0).float()
            total_correct += (
                predictions == batch_labels
            ).sum().item()

            total_examples += batch_size

        if total_examples:
            average_loss = total_loss / total_examples
            accuracy = total_correct / total_examples

            print(
                f"Epoch [{epoch + 1}/{epochs}] "
                f"Loss: {average_loss:.4f} "
                f"Accuracy: {accuracy:.4f}"
            )


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():
    if len(sys.argv) != 3:
        print(
            "Usage: python word2vec_skipgram.py "
            "<text_file> <epochs>"
        )
        sys.exit(1)

    text_file = sys.argv[1]

    try:
        epochs = int(sys.argv[2])
    except ValueError:
        sys.exit("Error: epochs must be an integer.")

    if epochs < 1:
        sys.exit("Error: epochs must be greater than zero.")

    # Prepare training data.
    vocabulary, couples, labels = process_data(
        text_file,
        WINDOW_SIZE,
    )

    vocabulary_size = len(vocabulary)

    print(f"Vocabulary size: {vocabulary_size}")
    print(f"Training pairs: {len(couples)}")
    print(f"Positive pairs: {sum(labels)}")
    print(f"Negative pairs: {len(labels) - sum(labels)}")

    # Select an available device.
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device: {device}")

    # Create and train the model.
    model = SkipGramModel(
        vocabulary_size,
        VECTOR_DIM,
    )

    train_model(
        model,
        couples,
        labels,
        epochs,
        device,
    )

    # Export learned target-word embeddings.
    save_embedding(
        "embedding.txt",
        model.target_embedding.weight,
        vocabulary,
    )

    # Save the model for later use.
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "vocabulary": vocabulary,
            "vector_dim": VECTOR_DIM,
            "window_size": WINDOW_SIZE,
        },
        "word2vec_skipgram.pt",
    )

    print("Model saved to word2vec_skipgram.pt")


if __name__ == "__main__":
    main()
