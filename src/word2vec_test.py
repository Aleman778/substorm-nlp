import numpy as np
from sklearn.manifold import TSNE
import torch
import torch.nn as nn
from data import word_sequence_from_korp_dataset
import matplotlib.pyplot as plt


def input_word(vocab):
    while True:
        word = input()
        try:
            word_idx = vocab.index(word)
            return word_idx
        except:
            print("Word `" + word + "` is not in the vocabulary")
    return 0
                

def main():
    # Load in the vocab
    # word_sequence = word_sequence_from_korp_dataset("../dataset/flashback_emails.csv")
    word_sequence = word_sequence_from_korp_dataset("../dataset/literaturbanken.csv")
    vocab = list(set(word_sequence))
    vocab_size = len(vocab)

    # Load in the model
    # filename = "flashback_word2vec_checkpoint.pt"
    filename = "literatur_word2vec_checkpoint.pt"
    embedding_size = 30
    model = nn.Linear(vocab_size, embedding_size, bias=False)
    model.load_state_dict(torch.load(filename, map_location=torch.device('cpu')))
    model.eval()
    
    input_embeddings = [None]*vocab_size
    for i, _ in enumerate(vocab):
        inputs = torch.Tensor(np.eye(vocab_size)[i])
        embedding = model(inputs)
        input_embeddings[i] = embedding.detach().numpy()

    # Plot t-SNE embeddings
    if False:
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

    # Test word-embeddings
    if True:
        while True:
            print("Vector algebra: Word1 - Word2 = Word3")
            print("Word1:")
            word1 = input_embeddings[input_word(vocab)];

            print("Word2:")
            word2 = input_embeddings[input_word(vocab)];

            vec3 = word1-word2
            shortest_dist = 1000
            word3 = "nothing found"
            for i, embed in enumerate(input_embeddings):
                dist = np.linalg.norm(embed-vec3)
                if dist < shortest_dist:
                    shortest_dist = dist
                    word3 = vocab[i]

            print("Word3:", word3, " (shortest distance:", shortest_dist, ")")

if __name__ == "__main__":
    main()
