#https://github.com/andreiburuntia/nltk-sentiment-analyzer/blob/master/
#add some modifications to fit python3 and fix encoding errors

import nltk
from nltk.classify import NaiveBayesClassifier
import sys,io


sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="gb18030")

# nltk.download('punkt')

def format_sentence(sent):
    return({word: True for word in nltk.word_tokenize(sent)})

pos = []
with open("./pos_tweets.txt",'r',encoding='utf-8') as f:
    for i in f: 
        pos.append([format_sentence(i), 'pos'])
 
neg = []
with open("./neg_tweets.txt",'r',encoding='utf-8') as f:
    for i in f: 
        neg.append([format_sentence(i), 'neg'])


# next, split labeled data into the training and test data
training = pos[:int((.8)*len(pos))] + neg[:int((.8)*len(neg))]
test = pos[int((.8)*len(pos)):] + neg[int((.8)*len(neg)):]

classifier = NaiveBayesClassifier.train(training)

print(classifier.classify(format_sentence("Iran vows to restart nuclear program if deal collapses")))