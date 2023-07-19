#spam detection filter

import nltk
nltk.download_shell()    #to download datasets necessary to operate in certain ways
msgs=[line.rstrip() for line in open('C:\\Users\\Ashraf\\Documents\\python-data-science-and-machine-learning-bootcamp-jose-portilla-master\\20-Natural-Language-Processing\\smsspamcollection\\SMSSpamCollection')]
for msg_no,msg in enumerate(msgs[:10]):
    print(msg_no,msg)
#we can say this is a tab(\t) seperated values file (tsv) where first column is a label if msg is spam or ham and second is msg itself
print(msgs[0])

import pandas as pd
msgs=pd.read_csv('C:\\Users\\Ashraf\\Documents\\python-data-science-and-machine-learning-bootcamp-jose-portilla-master\\20-Natural-Language-Processing\\smsspamcollection\\SMSSpamCollection',sep='\t',names=['label','messages'])
print(msgs.head())
print(msgs.groupby('label').describe())
msgs['length']=msgs['messages'].apply(len)
print(msgs.head())

import matplotlib.pyplot as plt
import seaborn as sns
sns.histplot(msgs['length'],bins=100)
plt.show()
#there are some messages which is very large
print(msgs['length'].describe())
#here max length is 910 which is absurd
print(msgs[msgs['length']==910]['messages'].iloc[0])#iloc to print out the entire string
msgs.hist(column='length',by='label',bins=60,figsize=(12,4))#pandas
plt.show()
#most of the ham msgs are between the langth 1 and 200 but spam msgs are in beetween 140 to 160

'''text preprocessing'''

import string
from nltk.corpus import stopwords

def text_process(mess):
    '''
    remove punctuations
    remove stop words, which are the words which are not helpful like is if all etc
    return list of clean text words
    '''
    nopunc=[char for char in mess if char not in string.punctuation]
    nopunc=''.join(nopunc)
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
'''there are words which are similar to each other like running ran run. Stemming helps to normalize the texts and returns the word run. Stemming neds reference dictionary
   , nltk comes with a lot of builtin datasets or references. But for this particular dataset it wont be much useful because we have shorthand words like U, Nah, dun, comp etc'''

#count vectorization

from sklearn.feature_extraction.text import CountVectorizer
bow_transformer=CountVectorizer(analyzer=text_process).fit(msgs['messages'])#bow is bag of words
print(len(bow_transformer.vocabulary_))#tot vocab words
mess4=msgs['messages'][3]
print(mess4)
bow4=bow_transformer.transform([mess4])
print(bow4)
print(bow_transformer.get_feature_names()[9554])
messages_bow=bow_transformer.transform(msgs['messages'])
print('shape of sparse matrix is',messages_bow.shape)#(rows,columns)
print(messages_bow.nnz)#non zero occurances

#normalisation
from sklearn.feature_extraction.text import TfidfTransformer #term frequency inverse document frequency, used for text processing
tfidf_transformer=TfidfTransformer().fit(messages_bow)
tfidf4=tfidf_transformer.transform(bow4)
print(tfidf4)# the weight value of each of these words vs the actual document
#print(tfidf_transformer.idf_(bow_transformer.vocabulary_['university']))# inverse document frequency
#converting entire bow corpus to tfidf corpus
messages_tfidf=tfidf_transformer.transform(messages_bow)

#naive bayes algo is good for classification like logistic reg
from sklearn.naive_bayes import MultinomialNB# Multinomial Naive Bayes
spam_detect_model=MultinomialNB().fit(messages_tfidf,msgs['label'])
print(spam_detect_model.predict(tfidf4)[0])
print(msgs['label'][3])
all_pred=spam_detect_model.predict(messages_tfidf)
print(all_pred)

from sklearn.model_selection import train_test_split
msg_train,msg_test,label_train,label_test=train_test_split(msgs['messages'],msgs['label'],test_size=0.3)
print(msg_train)

from sklearn.pipeline import Pipeline
pipeline=Pipeline([
    ('bow',CountVectorizer(analyzer=text_process)),
    ('tfidf',TfidfTransformer()),
    ('classifier',MultinomialNB())
])
'''
#using random forest classifier
from sklearn.ensemble import RandomForestClassifier
pipeline=Pipeline([
    ('bow',CountVectorizer(analyzer=text_process)),
    ('tfidf',TfidfTransformer()),
    ('classifier',RandomForestClassifier())
])
'''
pipeline.fit(msg_train,label_train)
predictions=pipeline.predict(msg_test)

from sklearn.metrics import classification_report
print(classification_report(label_test,predictions))
from sklearn.metrics import confusion_matrix
print(confusion_matrix(label_test,predictions))