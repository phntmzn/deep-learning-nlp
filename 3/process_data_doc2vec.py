# process_data_doc2vec.py

import re
import sys
from itertools import ngrams

import torch
from torch.utils.data import Dataset, DataLoader


def getLines(file_path):
    """Read non-empty lines from a text file."""
    with open(file_path, "r", encoding="utf-8") as file:
        return [line.rstrip() for line in file if line.strip()]


def normalize_word(word):
    """Lowercase a word and remove selected punctuation."""
    return re.sub(r"""[.,:;'"!?()]+""", "", word.lower())


def create_vocabulary(vocabulary, sentences):
    """Map each unique word to an integer ID."""
    vocabulary["<unk>"] = 0

    for sentence in sentences:
        for word in sentence.split():
            word = normalize_word(word)

            if word and word not in vocabulary:
                vocabulary[word] = len(vocabulary)


def process_data(textFile, window_size):
    """
    Generate context-target examples from documents.

    Returns:
        contexts: LongTensor [num_examples, window_size + 1]
                  First column contains the document ID.
        docids:   LongTensor [num_documents]
        targets:  LongTensor [num_examples]
        vocab:    Dictionary mapping words to integer IDs.
    """
    if window_size < 1:
        raise ValueError("window_size must be at least 1.")

    docs = getLines(textFile)

    vocab = {}
    create_vocabulary(vocab, docs)

    contexts = []
    docids = []
    targets = []

    with open("docs.legend", "w", encoding="utf-8") as file:
        for docid, sentence in enumerate(docs):
            file.write(f"{docid} {sentence}\n")
            docids.append(docid)

            words = [
                normalize_word(word)
                for word in sentence.split()
            ]
            words = [word for word in words if word]

            word_ngrams = list(ngrams(words, window_size))

            for i in range(len(word_ngrams) - 1):
                current_ngram = word_ngrams[i]
                target_word = word_ngrams[i + 1][0]

                context = [docid]

                for word in current_ngram:
                    context.append(vocab.get(word, vocab["<unk>"]))

                contexts.append(context)
                targets.append(vocab.get(target_word, vocab["<unk>"]))

    # Convert results directly into PyTorch tensors.
    if contexts:
        contexts_tensor = torch.tensor(
            contexts,
            dtype=torch.long,
        )
    else:
        contexts_tensor = torch.empty(
            (0, window_size + 1),
            dtype=torch.long,
        )

    docids_tensor = torch.tensor(docids, dtype=torch.long)
    targets_tensor = torch.tensor(targets, dtype=torch.long)

    return contexts_tensor, docids_tensor, targets_tensor, vocab


class ContextTargetDataset(Dataset):
    """Dataset for training a context-to-target prediction model."""

    def __init__(self, contexts, targets):
        if len(contexts) != len(targets):
            raise ValueError(
                "contexts and targets must have equal lengths."
            )

        self.contexts = contexts
        self.targets = targets

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return self.contexts[index], self.targets[index]


def main():
    if len(sys.argv) != 3:
        print("Usage: python process_data.py <text_file> <window_size>")
        sys.exit(1)

    text_file = sys.argv[1]
    window_size = int(sys.argv[2])

    contexts, docids, targets, vocab = process_data(
        text_file,
        window_size,
    )

    print("Vocabulary size:", len(vocab))
    print("Contexts shape:", tuple(contexts.shape))
    print("Document IDs shape:", tuple(docids.shape))
    print("Targets shape:", tuple(targets.shape))

    if len(targets) > 0:
        dataset = ContextTargetDataset(contexts, targets)

        loader = DataLoader(
            dataset,
            batch_size=32,
            shuffle=True,
        )

        batch_contexts, batch_targets = next(iter(loader))

        print("\nExample training batch:")
        print("Context shape:", tuple(batch_contexts.shape))
        print("Target shape:", tuple(batch_targets.shape))

    print("Document legend saved to docs.legend")


if __name__ == "__main__":
    main()
