# generator.py

import random

import torch
from torch.utils.data import Dataset, DataLoader


class ContextTargetDataset(Dataset):
    """Dataset containing target words, contexts, and labels."""

    def __init__(self, target, context, labels):
        self.target = torch.as_tensor(
            target, dtype=torch.long
        ).reshape(-1)

        self.context = torch.as_tensor(
            context, dtype=torch.long
        ).reshape(-1)

        self.labels = torch.as_tensor(
            labels, dtype=torch.float32
        ).reshape(-1)

        if not (
            len(self.target)
            == len(self.context)
            == len(self.labels)
        ):
            raise ValueError(
                "target, context, and labels must have equal lengths."
            )

        if len(self.target) == 0:
            raise ValueError("The dataset cannot be empty.")

    def __len__(self):
        return len(self.target)

    def __getitem__(self, index):
        target = self.target[index].reshape(1)
        context = self.context[index].reshape(1)
        label = self.labels[index].reshape(1)

        return (target, context), label


def generator(target, context, labels, batch_size):
    """
    Infinite generator that randomly samples examples.

    Yields:
        ([batch_target, batch_context], batch_labels)

    Each tensor has shape (batch_size, 1).
    """
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")

    dataset = ContextTargetDataset(target, context, labels)

    while True:
        indices = torch.randint(
            low=0,
            high=len(dataset),
            size=(batch_size,),
        )

        batch_target = dataset.target[indices].reshape(
            batch_size, 1
        )

        batch_context = dataset.context[indices].reshape(
            batch_size, 1
        )

        batch_labels = dataset.labels[indices].reshape(
            batch_size, 1
        )

        yield [batch_target, batch_context], batch_labels


def main():
    # Example data; replace these with your processed training data.
    target = [1, 2, 3, 4, 5]
    context = [10, 20, 30, 40, 50]
    labels = [1, 0, 1, 0, 1]

    batch_size = 3

    # Option 1: Use the generator directly.
    data_generator = generator(
        target,
        context,
        labels,
        batch_size,
    )

    (batch_target, batch_context), batch_labels = next(
        data_generator
    )

    print("Generator batch:")
    print("Target shape:", tuple(batch_target.shape))
    print("Context shape:", tuple(batch_context.shape))
    print("Labels shape:", tuple(batch_labels.shape))

    # Option 2: Use a PyTorch DataLoader.
    dataset = ContextTargetDataset(target, context, labels)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    (target_batch, context_batch), labels_batch = next(
        iter(loader)
    )

    print("\nDataLoader batch:")
    print("Target shape:", tuple(target_batch.shape))
    print("Context shape:", tuple(context_batch.shape))
    print("Labels shape:", tuple(labels_batch.shape))


if __name__ == "__main__":
    main()
