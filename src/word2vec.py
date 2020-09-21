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
        x = F.relu(self.embedding(x))
        return F.softmax(self.output(x), dim=0)

def create_skip_grams():
    

    
def main():
    # Hyper-parameters
    embedding_size = 100
    window_size = 2
    batch_size = 64
    num_epochs = 5
    learning_rate=0.01
    momentum=0.9

    # Create skip-grams
    sentences = data.get_sentences_from_korp_dataset

    dataset = data.KorpDataset("../dataset/flashback_emails.csv", window_size);
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
     # NOTE(alexander): embedding size should be smaller, otherwise what's the point just one-hot encode
    assert(embedding_size < dataset.vocab_size)
    
    # Create the word2vec model, optimizer and criterion
    model = Word2Vec(dataset.vocab_size, embedding_size)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    criterion = nn.MSELoss()

    # Train the model
    for epoch in range(num_epochs):
        epoch_loss = 0.0

        for batch_num, (inputs, targets) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward();
            optimizer.step();
            epoch_loss += loss.item()

        print('Epoch: %d - loss: %.3f' %
              (epoch + 1, epoch_loss / len(train_loader)))
    print('Finished Training')            

if __name__== "__main__":
    main()
