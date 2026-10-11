import re
from collections import Counter

import torch
from torch.nn.utils.rnn import pad_sequence

# 1. Sample documents
docs = [
    "Chuck Berry rolled over everyone who came before him ? and turned up "
    "[CA]everyone who came after. We'll miss you",
    "Help protect the progress we've made in helping millions of "
    "[CA]Americans get covered.",
    "Let's leave our children and grandchildren a planet that's healthier "
    "[CA]than the one we have today.",
    "The American people are waiting for Senate leaders to do their jobs.",
    "We must take bold steps now ? climate change is already impacting "
    "[CA]millions of people.",
    "Don't forget to watch Larry King tonight",
    "Ivanka is now on Twitter - You can follow her",
    "Last night Melania and I attended the Skating with the Stars Gala at "
    "[CA]Wollman Rink in Central Park",
    "People who have the ability to work should. But with the government "
    "[CA]happy to send checks",
    "I will be signing copies of my new book",
]

# 2. Convert documents to lowercase
docs = [doc.lower() for doc in docs]


# 3. Tokenizer equivalent to CountVectorizer's default token pattern.
#    It selects word tokens containing at least two characters.
def tokenize(text):
    return re.findall(r"(?u)\b\w\w+\b", text)


# 4. Build vocabulary from all documents.
tokenized_docs = [tokenize(doc) for doc in docs]

vocabulary_counts = Counter(
    token
    for document in tokenized_docs
    for token in document
)

# Match CountVectorizer's convention: vocabulary IDs are assigned
# in alphabetical order.
vocabulary = {
    token: index
    for index, token in enumerate(sorted(vocabulary_counts))
}

print(f"Vocabulary size: {len(vocabulary)}")


# 5. Convert each token to its integer vocabulary ID.
input_sequences = []

for document in tokenized_docs:
    sequence = [vocabulary[token] for token in document]
    input_sequences.append(sequence)


# 6. Find the longest sequence.
max_len = max(len(sequence) for sequence in input_sequences)

print(f"Maximum sequence length: {max_len}")


# 7. Pad sequences at the end with zeros.
#    Reserve 0 for padding by shifting vocabulary IDs up by 1.
input_tensors = [
    torch.tensor(
        [token_id + 1 for token_id in sequence],
        dtype=torch.long,
    )
    for sequence in input_sequences
]

input_array = pad_sequence(
    input_tensors,
    batch_first=True,
    padding_value=0,
)

print("Input shape:", tuple(input_array.shape))
print("\nFirst padded sequence:")
print(input_array[0])

print("\nVocabulary examples:")
for token, token_id in list(vocabulary.items())[:10]:
    print(f"{token}: {token_id + 1}")
