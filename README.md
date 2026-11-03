# Word2Vec and RNN from Scratch

This project implements Word2Vec using the Skip-gram with Negative Sampling (SGNS) approach and uses the learned word embeddings to initialize an RNN language model.

## Project Structure

```text
Word2Vec/
├── corpus.txt
├── preprocessing.py
├── W2V.py
├── README.md
├── word2vec_data.pt
├── word2vec_checkpoint.pt
└── rnn_checkpoint.pt
```

## 1. Data Preprocessing

- Tokenize and preprocess the text corpus.
- Build the vocabulary and word-to-ID mappings.
- Generate Skip-gram training pairs using a context window.
- Calculate the negative sampling distribution using word frequency raised to the power of 0.75.
- Add special tokens such as `<PAD>` and `<EOS>`.

## 2. Word2Vec

Implement the Skip-gram model with Negative Sampling (SGNS).

- Learn word embeddings using PyTorch.
- Use negative sampling to distinguish context words from randomly sampled negative words.
- Optimize the embeddings using Adam.

## 3. RNN Language Model

Use the pretrained Word2Vec embeddings to initialize an RNN language model.

- Initialize the embedding layer with pretrained Word2Vec weights.
- Use a recurrent neural network to predict the next word.
- Train the model with positive and negative sampling.
- Generate text autoregressively.

## Requirements

- Python 3.12+
- PyTorch
- NumPy

## Usage

Run the scripts in the following order:

```bash
python preprocessing.py
python W2V.py
python RNN.py
```

Make sure the required corpus is available in the project directory.

## Notes

Model checkpoints and generated data files (`.pt`, `.pth`) are not included in this repository. Run the