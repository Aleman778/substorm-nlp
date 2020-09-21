import pandas as pd
from nltk.tokenize import PunktSentenceTokenizer, RegexpTokenizer
from nltk.corpus import stopwords


def get_sentences_from_korp_dataset(filename):
    """Returns sentences from Korp datasets CSV file found at: https://spraakbanken.gu.se/korp/"""
    df = pd.read_csv(filename)
    num_sentences = df.shape[0]
    filter_words = set()
    sent_tokenizer = PunktSentenceTokenizer()
    word_tokenizer = RegexpTokenizer(r'\w+')
    sentences = 

    for i in range(0, num_sentences):
        text = str(df["left context"][i]) + " " + str(df["match"][i]) + " " + str(df["right_context"][i])
        sents = sent_tokenizer.tokenize(text);
        for sent in sents:
            word_tokens = word_tokenizer.tokenize(sent.lower())
            vocab = vocab.union(word_tokens)
            filtered_tokens = [w for w in word_tokens if not w in filter_words]
            sentences.append(filtered_tokens)
            self.total_num_words += len(filtered_tokens)
