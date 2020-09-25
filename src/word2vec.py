import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import data



# Word2Vec - SkipGram version
#  
#   Input (1-hot)    Hidden (embedding)    Output (softmax classifier)
#     0                                       0.0
#     0                0.3                    0.3
#     1        =>      0.1              =>    0.9
#     0                0.9                    0.4
#     0                                       0.1
#
# In the SkipGram version we train neural network to classify which words
# are related to the input word which in turn trains or word embedding.

class Word2Vec(nn.Module):
    def __init__(self, vocab_size, embedding_size):
        super(Word2Vec, self).__init__()
        self.embedding = nn.Linear(vocab_size, embedding_size, bias=False)
        self.output = nn.Linear(embedding_size, vocab_size, bias=False)

    def forward(self, x):
        x = self.embedding(x)
        return self.output(x)


def cuda_device_if_available():
    """Returns the CUDA device if it is available, 
    if no then the CPU device is returned instead.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print("Training on: " + device_name)
    return device


def main():
    # Hyper-parameters
    embedding_size = 30
    window_size = 2
    batch_size = 64
    num_epochs = 5
    learning_rate=0.001

    # Setup sequence of words and the vocabulary
    word_sequence = data.word_sequence_from_korp_dataset("../dataset/flashback_emails.csv")
    vocab = list(set(word_sequence))
    vocab_size = len(vocab)

    # Make skip-grams by combining center word and surrounding context words
    skip_grams = list()
    for i in range(1, len(word_sequence) - 1):
        center_word = vocab.index(word_sequence[i])
        for j in range(i - window_size, i + window_size):
            if j != i and j >= 0 and j < len(word_sequence):
                context_word = vocab.index(word_sequence[j])
                skip_grams.append((center_word, context_word))

    print("Some skip-gram examples:", [(vocab[i], vocab[j]) for (i, j) in skip_grams[:10]])

    # NOTE(alexander): embedding size should be smaller to get meaningful representation
    assert(embedding_size < vocab_size)


    # If CUDA support use that
    device = cuda_device_if_available()

    # Create the word2vec model, optimizer and criterion
    model = Word2Vec(vocab_size, embedding_size)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    lowest_loss = 1000
    filename = "checkpoint.pt"

    # Train the model
    for epoch in range(5000):
        # Create a mini batch
        input_batch = list()
        target_batch = list()
        indices_batch = np.random.choice(range(len(skip_grams)), batch_size, replace=False)
        for i in indices_batch:
            input_batch.append(np.eye(vocab_size)[skip_grams[i][0]]) # center word
            target_batch.append(skip_grams[i][1]) # context word

        inputs = torch.Tensor(input_batch)
        inputs = inputs.to(device)
        targets = torch.LongTensor(target_batch)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward();
        optimizer.step();
        epoch_loss = loss.item()

        if (epoch + 1) % 100 == 0:
            print("Epoch: %d - loss: %.6f" % (epoch + 1, epoch_loss))

        if epoch_loss < lowest_loss:
            torch.save(model.embedding.state_dict(), filename)
            lowest_loss = epoch_loss
            print("Reached lower loss of %.6f at epoch %d saving to %s" % (lowest_loss, epoch + 1, filename))
    print("Finished Training")

if __name__== "__main__":
    main()
