# doc2vec.py

import re
import sys
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from nltk.util import ngrams


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

WINDOW_SIZE = 3
VECTOR_DIM = 100
DEFAULT_EPOCHS = 1000
DEFAULT_BATCH_SIZE = 100
LEARNING_RATE = 0.001

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)


# ---------------------------------------------------------
# Text processing
# ---------------------------------------------------------

def normalize_word(word):
    """Lowercase a word and remove selected punctuation."""
    return re.sub(r"""[.,:;'"!?()]+""", "", word.lower())


def getLines(filename):
    """Read non-empty text lines from a file."""
    with open(filename, "r", encoding="utf-8") as file:
        return [line.rstrip() for line in file if line.strip()]


def create_vocabulary(vocabulary, docs):
    """Create a word-to-ID mapping."""
    vocabulary["<unk>"] = 0

    for doc in docs:
        for word in doc.strip().split():
            word = normalize_word(word)

            if word and word not in vocabulary:
                vocabulary[word] = len(vocabulary)


def process_data(text_file, window_size):
    """
    Convert documents into context windows and next-word targets.

    Context format:
        [document_id, word_1_id, word_2_id, word_3_id]

    Target:
        ID of the word following the context window.
    """
    docs = getLines(text_file)

    if not docs:
        raise ValueError(f"No documents found in {text_file!r}.")

    vocab = {}
    create_vocabulary(vocab, docs)

    contexts = []
    docids = []
    targets = []

    with open("docs.legend", "w", encoding="utf-8") as file:
        for docid, text in enumerate(docs):
            file.write(f"{docid} {text}\n")
            docids.append(docid)

            words = [
                normalize_word(word)
                for word in text.split()
            ]
            words = [word for word in words if word]

            word_ngrams = list(ngrams(words, window_size))

            for i in range(len(word_ngrams) - 1):
                context = [docid]

                for word in word_ngrams[i]:
                    context.append(vocab.get(word, vocab["<unk>"]))

                target_word = word_ngrams[i + 1][0]
                contexts.append(context)
                targets.append(vocab.get(target_word, vocab["<unk>"]))

    if not contexts:
        raise ValueError(
            "No training examples were created. "
            "Provide longer documents or reduce WINDOW_SIZE."
        )

    return (
        torch.tensor(contexts, dtype=torch.long),
        torch.tensor(docids, dtype=torch.long),
        torch.tensor(targets, dtype=torch.long),
        vocab,
    )


# ---------------------------------------------------------
# Batch generation
# ---------------------------------------------------------

class ContextDataset(Dataset):
    """Dataset for word contexts, document IDs, and targets."""

    def __init__(self, contexts, targets):
        self.contexts = contexts
        self.targets = targets

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        context = self.contexts[index]

        return (
            context[1],  # Word 1
            context[2],  # Word 2
            context[3],  # Word 3
            context[0],  # Document ID
        ), self.targets[index]


def generator(contexts, targets, batch_size):
    """Yield randomly shuffled batches indefinitely."""
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")

    dataset = ContextDataset(contexts, targets)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )

    while True:
        for inputs, batch_targets in loader:
            w1, w2, w3, docid = inputs

            yield (
                w1.to(DEVICE),
                w2.to(DEVICE),
                w3.to(DEVICE),
                docid.to(DEVICE),
            ), batch_targets.to(DEVICE)


# ---------------------------------------------------------
# Doc2Vec-style model
# ---------------------------------------------------------

class Doc2VecModel(nn.Module):
    """
    Predict a target word using three context-word embeddings
    and one document embedding.
    """

    def __init__(self, vocab_size, num_docs, vector_dim):
        super().__init__()

        self.word_embedding = nn.Embedding(
            vocab_size,
            vector_dim,
        )

        self.doc_embedding = nn.Embedding(
            num_docs,
            vector_dim,
        )

        self.output_layer = nn.Linear(
            vector_dim * 4,
            vocab_size,
        )

    def forward(self, w1, w2, w3, docid):
        # Each embedding has shape [batch_size, vector_dim].
        emb1 = self.word_embedding(w1)
        emb2 = self.word_embedding(w2)
        emb3 = self.word_embedding(w3)
        emb_doc = self.doc_embedding(docid)

        # Concatenate all four embeddings.
        context = torch.cat(
            [emb1, emb2, emb3, emb_doc],
            dim=1,
        )

        # Return logits. CrossEntropyLoss applies log-softmax.
        return self.output_layer(context)


# ---------------------------------------------------------
# Save document embeddings
# ---------------------------------------------------------

def save_embedding(output_file, weights, num_docs):
    """Save document vectors in a word2vec-style text format."""
    weights = weights.detach().cpu()

    with open(output_file, "w", encoding="utf-8") as file:
        file.write(f"{num_docs} {weights.shape[1]}\n")

        for index in range(num_docs):
            values = " ".join(
                f"{value:.6f}" for value in weights[index].tolist()
            )
            file.write(f"doc_{index} {values}\n")


# ---------------------------------------------------------
# Training
# ---------------------------------------------------------

def train_model(model, contexts, targets, epochs, batch_size):
    """Train the model using next-word prediction."""
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    criterion = nn.CrossEntropyLoss()

    batches = generator(contexts, targets, batch_size)
    steps_per_epoch = max(
        1,
        (len(targets) + batch_size - 1) // batch_size,
    )

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_examples = 0

        for _ in range(steps_per_epoch):
            (w1, w2, w3, docid), batch_targets = next(batches)

            optimizer.zero_grad()

            logits = model(w1, w2, w3, docid)
            loss = criterion(logits, batch_targets)

            loss.backward()
            optimizer.step()

            batch_size_actual = batch_targets.size(0)
            total_loss += loss.item() * batch_size_actual
            total_correct += (
                logits.argmax(dim=1) == batch_targets
            ).sum().item()
            total_examples += batch_size_actual

        average_loss = total_loss / total_examples
        accuracy = total_correct / total_examples

        print(
            f"Epoch {epoch + 1}/{epochs} "
            f"- Loss: {average_loss:.4f} "
            f"- Accuracy: {accuracy:.4f}"
        )


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():
    if len(sys.argv) < 2:
        print(
            "Usage: python doc2vec.py <text_file> [epochs] [batch_size]"
        )
        sys.exit(1)

    text_file = sys.argv[1]
    epochs = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_EPOCHS
    batch_size = (
        int(sys.argv[3]) if len(sys.argv) > 3
        else DEFAULT_BATCH_SIZE
    )

    if epochs < 1:
        raise ValueError("epochs must be at least 1.")

    contexts, docids, targets, vocab = process_data(
        text_file,
        WINDOW_SIZE,
    )

    vocab_size = len(vocab)
    num_docs = len(docids)

    print(f"Device: {DEVICE}")
    print(f"Documents: {num_docs}")
    print(f"Vocabulary size: {vocab_size}")
    print(f"Training examples: {len(targets)}")

    model = Doc2VecModel(
        vocab_size=vocab_size,
        num_docs=num_docs,
        vector_dim=VECTOR_DIM,
    ).to(DEVICE)

    print(model)

    train_model(
        model,
        contexts,
        targets,
        epochs=epochs,
        batch_size=batch_size,
    )

    save_embedding(
        "embedding_doc2vec.txt",
        model.doc_embedding.weight,
        num_docs,
    )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "vocab": vocab,
            "num_docs": num_docs,
            "vocab_size": vocab_size,
            "vector_dim": VECTOR_DIM,
            "window_size": WINDOW_SIZE,
        },
        "doc2vec_checkpoint.pt",
    )

    print("Document embeddings saved to embedding_doc2vec.txt")
    print("Model checkpoint saved to doc2vec_checkpoint.pt")


if __name__ == "__main__":
    main()
