
import re
import torch
import numpy as np
from collections import Counter


def build_dict():
    with open("corpus.txt", "r", encoding="utf-8") as f:
        text = f.read().lower()

    # Tokenize và tách câu theo từng dòng
    sentences = [
        re.findall(r"\b\w+\b", line)
        for line in text.splitlines()
    ]
    sentences = [s for s in sentences if s]

    # Vocabulary dùng chung cho Word2Vec và RNN
    tokens = [
        token
        for sentence in sentences
        for token in sentence
    ]

    freq = Counter(tokens)

    # Thêm special tokens
    freq["<PAD>"] = 0
    freq["<EOS>"] = len(sentences)

    vocab = sorted(freq.keys())

    word_to_id = {
        word: i
        for i, word in enumerate(vocab)
    }

    id_to_word = {
        i: word
        for word, i in word_to_id.items()
    }

    pad_id = word_to_id["<PAD>"]
    eos_id = word_to_id["<EOS>"]

    # Negative sampling probabilities
    # Không lấy PAD và EOS làm negative samples
    weights = np.array(
        [
            freq[word] ** 0.75
            if word not in ("<PAD>", "<EOS>")
            else 0.0
            for word in vocab
        ],
        dtype=np.float64
    )

    sample_probs = weights / weights.sum()

    # Training pairs cho Word2Vec (context window = 1)
    training_pairs = np.array([
        [word_to_id[sentence[i]], word_to_id[sentence[j]]]
        for sentence in sentences
        for i in range(len(sentence))
        for j in range(
            max(0, i - 1),
            min(len(sentence), i + 2)
        )
        if i != j
    ], dtype=np.int64).reshape(-1, 2)

    # Sentences cho RNN: thêm EOS vào cuối mỗi câu
    sentences_ids = [
        [
            word_to_id[token]
            for token in sentence
        ] + [eos_id]
        for sentence in sentences
    ]

    # Lưu dữ liệu
    torch.save({
        "word_to_id": word_to_id,
        "id_to_word": id_to_word,
        "training_pairs": torch.from_numpy(training_pairs),
        "sample_probs": torch.from_numpy(sample_probs),
        "sentences": sentences_ids,
        "vocab_size": len(vocab),
        "pad_id": pad_id,
        "eos_id": eos_id
    }, "../code/word2vec_data.pt")

    print("Đã lưu dữ liệu vào word2vec_data.pt")
    print("Vocabulary size:", len(vocab))
    print("PAD ID:", pad_id)
    print("EOS ID:", eos_id)
    print("Training pairs shape:", training_pairs.shape)
    print("Sample probabilities shape:", sample_probs.shape)
    print("Number of sentences:", len(sentences_ids))
    print("PAD probability:", sample_probs[pad_id])
    print("EOS probability:", sample_probs[eos_id])


if __name__ == "__main__":
    build_dict()
