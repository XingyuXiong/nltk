#https://github.com/andreiburuntia/nltk-sentiment-analyzer/blob/master/
#add some modifications to fit python3 and fix encoding errors

import nltk
from nltk.classify import NaiveBayesClassifier
import sys,io
import csv
import matplotlib.pyplot as plt


sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="gb18030")

# nltk.download('punkt')

def format_sentence(sent):
    return({word: True for word in nltk.word_tokenize(sent)})

pos = []
with open('./pos_tweets.txt','r',encoding='utf-8') as f:
    for i in f: 
        pos.append([format_sentence(i), 'pos'])
 
neg = []
with open('./neg_tweets.txt','r',encoding='utf-8') as f:
    for i in f: 
        neg.append([format_sentence(i), 'neg'])


# next, split labeled data into the training and test data
training = pos[:int((.8)*len(pos))] + neg[:int((.8)*len(neg))]
test = pos[int((.8)*len(pos)):] + neg[int((.8)*len(neg)):]

classifier = NaiveBayesClassifier.train(training)

#print(classifier.classify(
#print(format_sentence("Iran vows to restart nuclear program if deal collapses"))

#another way of sentiment analyzing
MAX_SCORE=5
predict_list=[[] for i in range(MAX_SCORE)]

read_data='../sentiment/data/amazon.csv'
with open(read_data,'r') as csv_file:
    reader=csv.reader(csv_file)
    next(csv_file)
    #same as:lines=csv_files.readlines()[1:]
    for row in reader:
        #print(format_sentence(row[0]))
        #print(row[1])
        predict_list[(int(row[1])-1)].append(classifier.classify(format_sentence(row[0])))


#draw
name_list=[str(i+1) for i in range(MAX_SCORE)]
print(name_list)
pos_count=[str(predict_list[i]).count('pos') for i in range(MAX_SCORE)]
neg_count=[str(predict_list[i]).count('neg') for i in range(MAX_SCORE)]
print(pos_count)
print(neg_count)
plt.bar(MAX_SCORE,pos_count,label='positive-comments',fc='b')
plt.bar(MAX_SCORE,neg_count,label='negative-comments',tick_label=name_list,fc='r')
plt.legend()
plt.show()