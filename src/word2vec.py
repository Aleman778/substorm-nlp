import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import random
import data

# Word2Vec - Skip-gram version
#  
#   Input (1-hot)    Hidden (embedding)    Output (softmax classifier)
#     0                                       0.0
#     0                0.3                    0.3
#     1        =>      0.1              =>    0.9
#     0                0.9                    0.4
#     0                                       0.1
#
# With negative sampling
#  
#   Input (1-hot)    Hidden (embedding)    Binary Classifier 0 or 1 
#     0                                       
#     0                0.3                    
#     1        =>      0.1              =>    0.9 (1 neighbour, 0 not related)
#     0                0.9                    
#     0                                       
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

class Word2VecNegSampling(nn.Module):
    def __init__(self, vocab_size, embedding_size):
        super(Word2VecNegSampling, self).__init__()
        self.embedding = nn.Linear(vocab_size, embedding_size, bias=False)
        self.output = nn.Linear(embedding_size, 1, bias=False)

    def forward(self, x):
        x = self.embedding(x)
        return self.output(x)

# Default hyper-parameters
embedding_size = 30
window_size = 2
batch_size = 64
num_epochs = 5000
learning_rate=0.001

# Dataset storage
word_sequence = None
vocab = None
vocab_size = None
skip_grams = None
num_skip_grams = None

# Training model
model = None
optimizer = None
criterion = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"

def setup_dataset(filename):
    global word_sequence, vocab, vocab_size, skip_grams, num_skip_grams
    # Setup sequence of words and the vocabulary
    word_sequence = data.word_sequence_from_korp_dataset("../dataset/" + filename)
    vocab = list(set(word_sequence))
    vocab_size = len(vocab)
    print("vocab_size:", vocab_size)

    # Make skip-grams by combining center word and surrounding context words
    skip_grams = list()
    for i in range(1, len(word_sequence) - 1):
        center_word = vocab.index(word_sequence[i])
        for j in range(i - window_size, i + window_size):
            if j != i and j >= 0 and j < len(word_sequence):
                context_word = vocab.index(word_sequence[j])
                skip_grams.append((center_word, context_word))
    num_skip_grams = len(skip_grams)

    print("Some skip-gram examples:", [(vocab[i], vocab[j]) for (i, j) in skip_grams[:10]])

    # NOTE(alexander): embedding size should be smaller to get meaningful representation
    assert(embedding_size < vocab_size)

def train_model(name, data_loader):
    print("Training on: " + device_name)
    lowest_loss = 1000
    filename = name + "_checkpoint.pt"

    # Train the model
    for epoch in range(num_epochs):
        # Create a mini batch
        inputs, targets = data_loader()
        inputs = inputs.to(device)
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

def load_samples():
    input_batch = list()
    target_batch = list()
    indices_batch = np.random.choice(range(num_skip_grams), batch_size, replace=False)
    for i in indices_batch:
        input_batch.append(np.eye(vocab_size)[skip_grams[i][0]]) # center word
        target_batch.append(skip_grams[i][1]) # context word

    inputs = torch.Tensor(input_batch)
    targets = torch.LongTensor(target_batch)
    return inputs, targets

def load_negative_samples():
    input_batch = list()
    target_batch = list()

    center_word = random.randint(0, vocab_size - 1)
    positive_samples = []
    negative_samples = []
    for sg in skip_grams:
        if sg[0] == center_word: 
            continue
        if sg[1] == center_word:
            positive_samples.append(sg[1])
        else:
            negative_samples.append(sg[1])

    positive_sample = random.randint(0, len(positive_samples) - 1)
    input_batch.append(np.eye(vocab_size)[skip_grams[positive_sample][0]]) # target skipgram
    target_batch.append([0])

    negative_samples = np.random.choice(negative_samples, batch_size, replace=False)
    for i in negative_samples:
        input_batch.append(np.eye(vocab_size)[i]) # center word
        target_batch.append([1])

    # print("Center word:", vocab[center_word])
    # print("Positive sample:", vocab[positive_sample])
    # print("Negative samples:")
    # for sample in negative_samples:
        # print("-", vocab[sample]);

    inputs = torch.Tensor(input_batch)
    targets = torch.Tensor(target_batch)
    return inputs, targets

def test_flashback_word2vec():
    global model, optimizer, criterion

    # Setup dataset
    setup_dataset("flashback_emails.csv");

    # Create the word2vec model, optimizer and criterion
    model = Word2Vec(vocab_size, embedding_size)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    train_model("flashback_word2vec", load_samples);

def test_flashback_word2vec_neg_sampling():
    global model, optimizer, criterion, batch_size

    # Setup dataset
    setup_dataset("flashback_emails.csv");

    # Use smaller batch size for negative sampling
    batch_size = 20

    # Create the word2vec model, optimizer and criterion
    model = Word2VecNegSampling(vocab_size, embedding_size)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss()
    train_model("flashback_word2vec", load_negative_samples);

def test_literatur_word2vec():
    global model, optimizer, criterion

    # Setup dataset
    setup_dataset("literaturbanken.csv");

    # Create the word2vec model, optimizer and criterion
    model = Word2Vec(vocab_size, embedding_size)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    train_model("literatur_word2vec", load_samples);

def test_literatur_word2vec_neg_sampling():
    global model, optimizer, criterion, batch_size

    # Setup dataset
    setup_dataset("literaturbanken.csv");

    # Use smaller batch size for negative sampling
    batch_size = 5

    # Create the word2vec model, optimizer and criterion
    model = Word2VecNegSampling(vocab_size, embedding_size)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss()
    train_model("literatur_word2vec", load_negative_samples);


if __name__== "__main__":
    # test_flashback_word2vec()
    # test_flashback_word2vec_neg_sampling()
    # test_literatur_word2vec()
    test_literatur_word2vec_neg_sampling()
