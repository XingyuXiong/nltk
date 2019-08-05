import nltk
import pandas as pd
import re


def parse_document(document):
    document=re.sub('\n',' ',document)
    if isinstance(document,str):
        document=document
    else:
        raise ValueError('Document is not string!')
    document=document.strip()
    sentences=nltk.sent_tokenize(document)
    sentences=[sentence.strip() for sentence in sentences]
    return sentences


# sample document
text = """
FIFA was founded in 1904 to oversee international competition among the national associations of Belgium, 
Denmark, France, Germany, the Netherlands, Spain, Sweden, and Switzerland. Headquartered in Zürich, its 
membership now comprises 211 national associations. Member countries must each also be members of one of 
the six regional confederations into which the world is divided: Africa, Asia, Europe, North & Central America 
and the Caribbean, Oceania, and South America.
"""


sentences=parse_document(text)
tokenized_sentences=[nltk.word_tokenize(sentence) for sentence in sentences]

tagged_sentences=[nltk.pos_tag(sentence) for sentence in tokenized_sentences]
ne_chunked_sents=[nltk.ne_chunk(tagged) for tagged in tagged_sentences]

named_entites=[]
for ne_tagged_sentence in ne_chunked_sents:
    for tagged_tree in ne_tagged_sentence:
        if hasattr(tagged_tree,'label'):
            entity_name=' '.join(c[0] for c in tagged_tree.leaves())
            entity_type=tagged_tree.label()
            named_entites.append((entity_name,entity_type))
            named_entites=list(set(named_entites))


entity_frame=pd.DataFrame(named_entites,columns=['Entity Name','Entity Type'])

print(entity_frame)