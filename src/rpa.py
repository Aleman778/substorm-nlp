#!/usr/bin/env python
# -*- coding: utf-8 -*-

import nltk
from nltk import sent_tokenize, word_tokenize
from nltk.tokenize import PunktSentenceTokenizer


pos_tag_list = """POS tag list:
  CC	coordinating conjunction
  CD	cardinal digit
  DT	determiner
  EX	existential there (like: "there is" ... think of it like "there exists")
  FW	foreign word
  IN	preposition/subordinating conjunction
  JJ	adjective	'big'
  JJR	adjective, comparative	'bigger'
  JJS	adjective, superlative	'biggest'
  LS	list marker	1)
  MD	modal	could, will
  NN	noun, singular 'desk'
  NNS	noun plural	'desks'
  NNP	proper noun, singular	'Harrison'
  NNPS	proper noun, plural	'Americans'
  PDT	predeterminer	'all the kids'
  POS	possessive ending	parent\'s
  PRP	personal pronoun	I, he, she
  PRP$	possessive pronoun	my, his, hers
  RB	adverb	very, silently,
  RBR	adverb, comparative	better
  RBS	adverb, superlative	best
  RP	particle	give up
  TO	to	go 'to' the store.
  UH	interjection	errrrrrrrm
  VB	verb, base form	take
  VBD	verb, past tense	took
  VBG	verb, gerund/present participle	taking
  VBN	verb, past participle	taken
  VBP	verb, sing. present, non-3d	take
  VBZ	verb, 3rd person sing. present	takes
  WDT	wh-determiner	which
  WP	wh-pronoun	who, what
  WP$	possessive wh-pronoun	whose
  WRB	wh-abverb	where, when"""


def main():
    query = "Maila alla studenter från LTU att jag är sjuk. Ses imorgon kl 8 istället."

    # sentences = sent_tokenize(query)
    # for sentence in sentences:
        # tokens = word_tokenize(sentence)
        # if (tokens[len(tokens) - 1] != '.'):
            # print("error: query must end with a period")
        # print(tokens)
        
    tokenized = word_tokenize(query);
    print("Sv:", nltk.pos_tag(tokenized))

    query = "Mail all students from LTU that I am sick. See you tomorrow 8am instead."
    tokenized = word_tokenize(query)
    print("Eng:", nltk.pos_tag(tokenized))
    print(pos_tag_list)
    
if __name__ == "__main__":
    main()

