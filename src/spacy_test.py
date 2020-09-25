import spacy

# Load English tokenizer, tagger, parser, NER and word vectors
nlp = spacy.load("en_core_web_sm")

# Process whole documents
text = ("Send email to alemen-6@student.ltu.se saying Hello World!")
doc = nlp(text)

# Analyze syntax
print("Noun phrases:", [chunk.text for chunk in doc.noun_chunks])
print("Verbs:", [token.lemma_ for token in doc if token.pos_ == "VERB"])

print("Doc:", doc)

# Find named entities, phrases and concepts
for entity in doc.ents:
    print(entity.text, entity.label_)

for token in doc:
    print(token.text, token.has_vector, token.vector_norm, token.is_oov)
