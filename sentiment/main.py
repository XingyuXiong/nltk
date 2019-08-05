import nltk.sentiment.vader as vader
import csv
import os
import numpy as np
import matplotlib.pyplot as plt


MAX_SCORE=5

sid=vader.SentimentIntensityAnalyzer()

def rating(com,pos,neu,neg,maxscore):
    return com


if __name__=='__main__':
    csv_file=csv.reader(open('./data/movie.csv'))
    data_list={}
    i=0
    for row in csv_file:
        data_list[i]=row
        i+=1
    pol_scores=[sid.polarity_scores(data_list[i][0]) for i in range(1,len(data_list))]
    rate_scores=[rating(dict['compound'],dict['pos'],dict['neu'],dict['neg'],MAX_SCORE) for dict in pol_scores]
    user_scores=[int(data_list[i][1]) for i in range(1,len(data_list))]
    corr=np.corrcoef(rate_scores,user_scores)

    print_mat=[[] for i in range(MAX_SCORE)]
    for i in range(1,len(user_scores)+1):
        print_mat[int(data_list[min(i,len(user_scores)-1)][1])-1].append(rate_scores[i-1])
    #print(print_mat)
    plt.boxplot(x=print_mat)
    plt.show()
