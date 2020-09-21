import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.functional as F
import matplotlib.pyplot as plt

# Word2Vec - SkipGram version
#  
#   Input (1-hot)    Hidden (embedding)    Output (softmax classifier)
#     0                                       0.0
#     0                0.3                    0.3
#     1        =>      0.1              =>    0.0
#     0                0.9                    0.4
#     0                                       0.1
#
# In the SkipGram version we train neural network to classify which words
# are related to the input word which in turn trains or word embedding.

class Word2VecSkipGram(nn.Module):
    def __init__(self, vocab_size, embedding_size):
        super(Word2VecSkipGram, self).__init__()
        self.embedding = nn.Linear(vocab_size, embedding_size)
        self.output = nn.Linear(embedding_size, vocab_size)

    def forward(self, x):
        x = self.embedding(x);
        return F.softmax(self.oux);

