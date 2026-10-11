# generating_training_batches_for_doc2vec.py

import torch
from torch.utils.data import Dataset, DataLoader


class ContextTargetDataset(Dataset):
    """Dataset containing context words, document IDs, and target words."""

    def __init__(self, contexts, targets):
        self.contexts = torch.as_tensor(contexts, dtype=torch.long)
        self.targets = torch.as_tensor(targets, dtype=torch.long)

        if len(self.contexts) != len(self.targets):
            raise ValueError("contexts and targets must have the same length.")

        if self.contexts.ndim != 2 or self.contexts.shape[1] < 4:
            raise ValueError(
                "contexts must have shape [num_examples, 4 or more]: "
                "[docid, w1, w2, w3, ...]"
            )

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        context = self.contexts[index]

        docid = context[0]
        w1 = context[1]
        w2 = context[2]
        w3 = context[3]
        target = self.targets[index]

        return (w1, w2, w3, docid), target


def generator(contexts, targets, batch_size):
    """
    Yield randomly shuffled batches indefinitely.

    Returns:
        Inputs: (w1, w2, w3, docid)
        Targets: target word IDs
    """
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")

    dataset = ContextTargetDataset(contexts, targets)

    if len(dataset) == 0:
        raise ValueError("Cannot generate batches from an empty dataset.")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )

    while True:
        for inputs, batch_targets in loader:
            w1, w2, w3, docid = inputs
            yield (w1, w2, w3, docid), batch_targets


# -------------------------------------------------------------------
# Example usage
# -------------------------------------------------------------------
#
# from process_data import process_data
# from generator import generator
#
# contexts, docids, targets, vocab = process_data(
#     "sample.txt",
#     window_size=3,
# )
#
# batch_size = 32
# batches = generator(contexts, targets, batch_size)
#
# (inputs, batch_targets) = next(batches)
# w1, w2, w3, docid = inputs
#
# print("w1 shape:", w1.shape)
# print("w2 shape:", w2.shape)
# print("w3 shape:", w3.shape)
# print("docid shape:", docid.shape)
# print("targets shape:", batch_targets.shape)
          
