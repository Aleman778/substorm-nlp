from spacy.pipeline import EntityRuler
import spacy
from spacy import displacy

# Load English tokenizer, tagger, parser, NER and word vectors


nlp = spacy.load("en_core_web_md")
ruler = EntityRuler(nlp)
pattern = [{"label": "EMAIL", "pattern": {"REGEX": "[a-z0-9\.\-+_]+ *@[a-z0-9\.\-+_]+"}}]
ruler.add_patterns(pattern)
nlp.add_pipe(ruler)






# Process whole documents
# text = ("Send an email to alemen-6@student.ltu.se saying Hello World!")
text = ("Hello World! Send this to Alexander Mennborg alemen-6@student.ltu at 10 pm today.")
doc = nlp(text)


# Analyze syntax
print("Noun phrases:", [chunk.text for chunk in doc.noun_chunks])
print("Verbs:", [token.lemma_ for token in doc if token.pos_ == "VERB"])
print("Doc:", doc)



# displacy.serve(doc, style="dep")
displacy.serve(doc, style="ent")
