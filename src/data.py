import pandas as pd
from nltk.tokenize import PunktSentenceTokenizer, RegexpTokenizer
from nltk.corpus import stopwords


def word_sequence_from_korp_dataset(filename, filter_stopwords=True):
    """Returns sentences from Korp datasets CSV file found at: https://spraakbanken.gu.se/korp/"""
    df = pd.read_csv(filename)
    num_sentences = df.shape[0]
    filter_words = set()
    if filter_stopwords:
        filter_words = filter_words.union(stopwords.words("swedish"));
    sent_tokenizer = PunktSentenceTokenizer()
    word_tokenizer = RegexpTokenizer(r'\w+')
    word_sequence = list()

    for i in range(0, num_sentences):
        text = str(df["left context"][i]) + " " + str(df["match"][i]) + " " + str(df["right_context"][i])
        sents = sent_tokenizer.tokenize(text);
        for sent in sents:
            word_tokens = word_tokenizer.tokenize(sent.lower())
            filtered_tokens = [w for w in word_tokens if not w in filter_words]
            word_sequence.extend(filtered_tokens)
    return word_sequence
