#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Idriss
"""
import time
import pandas as pd

import os
os.chdir('/home/idm/Downloads')
data = pd.read_csv('data.csv',sep = ',',encoding = 'utf-8')

from sklearn.feature_extraction.text import TfidfVectorizer
        
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize,pos_tag

lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

def stem_lem_tokens(tokens, stemmer):
     lemmatized = []
     for item in tokens:
         stem_ = stemmer.stem(item)
         lemmatized.append(lemmatizer.lemmatize(stem_))
     return lemmatized

def tokenize(text):
     tokens = word_tokenize(text)
     lemmatized = stem_lem_tokens(tokens, stemmer)
     return lemmatized

keywords = ['bug','fix','problem','issue','defect','crash','solve','but','before','now','never','always',
            'correct','error','behavior','performance','add','please','could','would','hope','improve',
            'miss','need','prefer','request','should','suggest','want','wish','must change','future','enhance',
            'help','support','assist','when','situation','did','when','while','experience','great','good',
            'nice','very','cool','love','hate','bad','worst','too','up','awesome','down','like',
            'superb','fast','excellent','perfect']

tdif = TfidfVectorizer(tokenizer = word_tokenize,ngram_range=(1,2),stop_words = 'english',max_features = 100,vocabulary = set(keywords),use_idf = False,lowercase = False )
temp2 = tdif.fit_transform(data.reviewText)

count_names = tdif.get_feature_names()
data_CV = pd.DataFrame(temp2.todense(), index=data.index, columns=count_names)

tdif = TfidfVectorizer(tokenizer = tokenize,ngram_range=(1,2),stop_words = 'english',max_features = 50,use_idf = False,lowercase = False )
temp2 = tdif.fit_transform(data.reviewText)

count_names = tdif.get_feature_names()
data_CV2 = pd.DataFrame(temp2.todense(), index=data.index, columns=count_names)

data_CV_concat = pd.concat([data_CV.reset_index(drop = True),data_CV2.reset_index(drop = True)],axis = 1)


liste = ['pos_emojis', 'neg_emojis', 'neutral_emojis','exclamation',
        'interrogation', 'Base form', 'Present', 'Present participle', 'Past',
        'Past participle', 'Modal', 'future',  'Sentiment','UE','BR','FR']

for col in liste:
	data_CV_concat[col] = data[col]


data_CV_concat['reviewText'] = data['reviewText']

data_CV_concat["Length"] = data_CV_concat["reviewText"].apply( lambda x: len(x.split()))

data_CV_concat = data_CV_concat[data_CV_concat["Length"] > 20].reset_index(drop=True)


import numpy as np
import matplotlib
from matplotlib import pyplot as plt

matplotlib.use('tkAgg')
try:
    from sklearn.model_selection import train_test_split
except ImportError:
    from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler

# libact classes
from libact.base.dataset import Dataset
from libact.query_strategies.multilabel import MultilabelWithAuxiliaryLearner
from libact.models.multilabel import BinaryRelevance
from libact.models import LogisticRegression,SVM


def run(data,trn_ds, qs, quota):
    #data needs to have the same index as trn_ds but with reviewText column
    for _ in range(quota):
        # Standard usage of libact objects
        t1 = time.time()
        ask_id = qs.make_query()
        t2 = time.time()
        print('query took ',t2 - t1)
        lb = input('What is the label of the '+str(ask_id) +' review? ' + str(data['reviewText'].iloc[ask_id])+' ')
        
        t3 = time.time()
        trn_ds.update(ask_id, eval(lb))
        t4 = time.time()
        print('update took ',t4-t3)
        
        #We want to save the labels added
        data['UE'].iloc[ask_id] = eval(lb)[0]
        data['BR'].iloc[ask_id] = eval(lb)[1]
        data['FR'].iloc[ask_id] = eval(lb)[2]
        t5 = time.time()
        
        print('saving labels took ',t5 - t4)
        
from itertools import combinations

labels = ['UE','BR','FR']
to_remove = ['reviewText']

X = data_CV_concat.drop(labels + to_remove,axis = 1)
X = StandardScaler().fit_transform(X)

Y = data_CV_concat[labels].reset_index(drop=True)

labeled_entries = Y['UE'].value_counts()[0] + Y['UE'].value_counts()[1]

threshold = 20

'''
Balanced test set
'''

indices = []

#Combination of 2 labels
for elem in combinations(['UE','BR','FR'],2):
    indices.extend(np.array(Y[(Y[elem[0]] == 1) & (Y[elem[1]] == 1)].index[:threshold]))


for elem in combinations(['UE','BR','FR'],1):
    indices.extend(np.array(Y[(Y[elem[0]] == 1)].index[:threshold]))


test_index = list(set(indices))


'''
Randomly Selected Test set
'''
#ratio_test = 0.2
#len_test = int(labeled_entries*ratio_test)
#
#np.random.seed(1367)
#
#
#test_index = np.random.choice(range(labeled_entries),len_test,False)

'''
'''

data_shape = len(X)
train_index = list(set(range(data_shape)).difference(test_index))

labeled_train = list(set(range(labeled_entries)).difference(test_index))

trn_ds = Dataset(X[train_index], Y.iloc[labeled_train].values.tolist() + [None]*(len(train_index) - len(labeled_train)))
tst_ds = Dataset(X[test_index],Y.iloc[test_index].values.tolist())


data_CV_train = data_CV_concat.iloc[train_index]

'''
MAIN FUNCTION
'''

result = {'Hamming': [],'F1': []}
    
model = BinaryRelevance(LogisticRegression())

quota = 20  # number of samples to query


#EXECUTE FROM HERE FOR ITERATIONS

qs1 = MultilabelWithAuxiliaryLearner(
trn_ds,
BinaryRelevance(LogisticRegression()),
BinaryRelevance(SVM()),
criterion='hlr')

run(data_CV_train,trn_ds, qs1, quota)

model.train(trn_ds)

X , y = zip(*tst_ds.get_labeled_entries())

pred = model.predict(X)

output = pd.DataFrame()
output['UE_pred'] = [pred[i][0] for i in range(len(pred))]
output['BR_pred'] = [pred[i][1] for i in range(len(pred))]
output['FR_pred'] = [pred[i][2] for i in range(len(pred))]

true = Y.iloc[test_index].reset_index(drop=True)

output['reviewText'] = np.array(data_CV_concat.iloc[test_index]['reviewText'])

output = pd.concat([output,true],axis = 1)

output.to_csv('output_test.csv')

score_hamming = model.score(tst_ds, criterion='hamming')
score_f1 = model.score(tst_ds,criterion = 'f1')

result['Hamming'].append(score_hamming)
result['F1'].append(score_f1)  

data.iloc[train_index][['UE','BR','FR']] = data_CV_train[['UE','BR','FR']]
data.to_csv('data_new.csv')

print('The result of Hamming scores is ',result['Hamming'])
print('The result of F1 scores is ',result['F1'])





'''
AFTER FINISHING ALL ITERATIONS
'''

'''
Hamming Loss
'''
iterations_num = np.arange(1, len(result['Hamming']) + 1)
fig = plt.figure(figsize=(9, 6))
ax = plt.subplot(111)
plt.plot(iterations_num, result['Hamming'], 'r', label='AuxiliaryLearner_hlr')

box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.75, box.height])
plt.legend(loc=2, bbox_to_anchor=(1.05, 1), borderaxespad=0.)
plt.xlabel('Iteration index')
plt.ylabel('Loss')
plt.title('Experiment Result (Hamming Loss)')
plt.show()


'''
F1 Loss
'''
iterations_num = np.arange(1, len(result['F1']) + 1)
fig = plt.figure(figsize=(9, 6))
ax = plt.subplot(111)
plt.plot(iterations_num, result['F1'], 'r', label='AuxiliaryLearner_hlr')

box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.75, box.height])
plt.legend(loc=2, bbox_to_anchor=(1.05, 1), borderaxespad=0.)
plt.xlabel('Iteration index')
plt.ylabel('Loss')
plt.title('Experiment Result (F1 Loss)')
plt.show()
#
#if __name__ == '__main__':

#    
#    main()




