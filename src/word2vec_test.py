import numpy as np
from sklearn.manifold import TSNE
import torch
import torch.nn as nn
from data import word_sequence_from_korp_dataset
import matplotlib.pyplot as plt


def main():
    # Load in the vocab
    word_sequence = word_sequence_from_korp_dataset("../dataset/flashback_emails.csv")
    vocab = list(set(word_sequence))
    vocab_size = len(vocab)

    # Load in the model
    filename = "checkpoint.pt"
    embedding_size = 30
    model = nn.Linear(vocab_size, embedding_size, bias=False)
    model.load_state_dict(torch.load(filename, map_location=torch.device('cpu')))
    model.eval()
    
    input_embeddings = [None]*vocab_size
    for i, _ in enumerate(vocab):
        inputs = torch.Tensor(np.eye(vocab_size)[i])
        embedding = model(inputs)
        input_embeddings[i] = embedding.detach().tolist()

    tsne_embeddings = TSNE(n_components=2).fit_transform(np.array(input_embeddings))
    x = [0]*vocab_size
    y = [0]*vocab_size
    for i, embd in enumerate(tsne_embeddings):
        x[i] = embd[0]
        y[i] = embd[1]
        
    fig, ax = plt.subplots()
    ax.scatter(x, y)
    for i, word in enumerate(vocab):
        ax.annotate(word, (x[i] + 0.1, y[i] + 0.1))
    plt.show()

if __name__ == "__main__":
    main()
