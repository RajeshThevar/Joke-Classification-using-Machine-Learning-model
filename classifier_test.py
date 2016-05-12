import matplotlib.pyplot as plt
import csv
from textblob import TextBlob
import pandas
import sklearn
import sys
import cPickle
import time
import random
import nltk
import numpy as np
from nltk.corpus import wordnet
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC, LinearSVC
from sklearn import svm, grid_search, datasets
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedKFold, cross_val_score, train_test_split 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.learning_curve import learning_curve
from nltk.corpus import stopwords
from collections import Counter
from nltk import word_tokenize, WordNetLemmatizer

messages = [line.rstrip() for line in open('JokeNonJokecollection/Collection')]
print '\nNumber of training data\n'
print len(messages)

for message_no, message in enumerate(messages[:10]):
    print '\nprinting first 10 sentence for output\n'
    print message_no, message
    

messages = pandas.read_csv('JokeNonJokecollection/Collection', sep='\t', quoting=csv.QUOTE_NONE, names=["label", "message"])
#print messages

print '\nlabel and headers of the training data available\n'
print(messages.groupby('label').describe())

messages['length'] = messages['message'].map(lambda text: len(text))
#print messages.head()

#plt.imshow(messages.length.plot(bins=20, kind='hist'))
#plt.show()

#print messages.length.describe()
#print list(messages.message[messages.length > 900])


def split_into_tokens(message):
    message = unicode(message, 'utf8')
    return TextBlob(message).words
    
#print messages.message.head()
print '\nmessages are tokenized before classification approach\n'
print messages.message.head().apply(split_into_tokens)


#print TextBlob("Hello world, how is it going").tags
print '\nmessages is tagged into part of speech before classification approach\n'
print TextBlob("Today a man knocked on my door and asked for a small donation towards the local swimming pool.I gave him a glass of water.").tags


def split_into_lemmas(message):
    message = unicode(message, 'utf8').lower()
    words = TextBlob(message).words
    return [word.lemma for word in words]
    

print '\nmessages is split into lemmas before classification approach\n'
print messages.message.head().apply(split_into_lemmas)

stoplist = stopwords.words('english')
#random.shuffle(messages)
all_messages = nltk.FreqDist(messages)

#synonyms = []
#antonyms = []
#
#for syn in wordnet.synsets("message"):
#    for l in syn.lemmas():
#        synonyms.append(l.name())
#        if l.antonyms():
#            antonyms.append(l.antonyms()[0].name())
# 

#def preprocess(message):
#    lemmatizer = WordNetLemmatizer()
#    return [lemmatizer.lemmatize(word.lower()) for word in word_tokenize(unicode(messages, errors='ignore'))]
# 
#
#def get_features(message, setting):
#    if setting=='bow':
#        return {word: count for word, count in Counter(preprocess(messages)).items() if not word in stoplist}
#    else:
#        return {word: True for word in preprocess(messages) if not word in stoplist}
# 
bow_transformer = CountVectorizer(analyzer=split_into_lemmas).fit(messages['message'])
#print len(bow_transformer.vocabulary_)

message4 = messages['message'][3]
print message4

bow4 = bow_transformer.transform([message4])
print '\nBag of words approach for the above sentence, example output\n'
print bow4
#print bow4.shape

#print bow_transformer.get_feature_names()[6736]
#print bow_transformer.get_feature_names()[8013]


messages_bow = bow_transformer.transform(messages['message'])
#print 'sparse matrix shape:', messages_bow.shape
#print 'number of non zeros:', messages_bow.nnz
#print 'sparsity: %.2f%%' % (100.0 * messages_bow.nnz / (messages_bow.shape[0] * messages_bow.shape[1]))

tfidf_transformer = TfidfTransformer().fit(messages_bow)
tfidf4 = tfidf_transformer.transform(bow4)
print '\nTfIdf of the above sentence for example\n'
print tfidf4

#print tfidf_transformer.idf_[bow_transformer.vocabulary_['u']]
#print tfidf_transformer.idf_[bow_transformer.vocabulary_['university']]

messages_tfidf = tfidf_transformer.transform(messages_bow)
#print messages_tfidf.shape

#start_time = time.clock()
spam_detector = MultinomialNB().fit(messages_tfidf, messages['label'])

#print time.clock() - start_time, "seconds"


#print 'predicted:', spam_detector.predict(tfidf4)[0]
#print 'expected:', messages.label[3]


all_predictions = spam_detector.predict(messages_tfidf)
print '\nPredicting all sentences\n'
print all_predictions


#print 'accuracy', accuracy_score(messages['label'], all_predictions)
#print 'confusion matrix\n', confusion_matrix(messages['label'], all_predictions)
#print '(row=expected, col=predicted)'


#plt.matshow(confusion_matrix(messages['label'], all_predictions), cmap=plt.cm.binary, interpolation='nearest')
#plt.title('confusion matrix')
#plt.colorbar()
#plt.ylabel('expected label')
#plt.xlabel('predicted label')

#print classification_report(messages['label'], all_predictions)


msg_train, msg_test, label_train, label_test = \
train_test_split(messages['message'], messages['label'], test_size=0.2)

print '\ntest message size',len(msg_test), 'train message size',len(msg_train), 'total message size',len(msg_train) + len(msg_test)

pipeline = Pipeline([
            ('bow', CountVectorizer(analyzer=split_into_lemmas)),
            ('tfidf', TfidfTransformer()),
            ('Classifier', MultinomialNB()),
            ])
            
scores = cross_val_score(pipeline,
                        msg_train,
                        label_train,
                        cv=10,
                        scoring='accuracy',
                        n_jobs=-1,
                        )
                        
#print scores
#print 'mean', scores.mean(), 'standard deviation',scores.std()


params = {
    'tfidf__use_idf': (True, False),
    'bow__analyzer': (split_into_lemmas, split_into_tokens),
}

grid = GridSearchCV(
    pipeline,  
    params,  
    refit=True, 
    n_jobs=-1, 
    scoring='accuracy',  
    cv=StratifiedKFold(label_train, n_folds=5),  
)   

nb_detector = grid.fit(msg_train, label_train)
#print nb_detector.grid_scores_


#print nb_detector.predict_proba(['Hi Mom, How are you?'])[0]
#print nb_detector.predict_proba(["WINNER! Credit for Free @ Lexington Corporation of 100000$!!"])[0]

print 'We can see the probability with which the sentence is judged, sentence is joke'
print nb_detector.predict_proba(['Today a man knocked on my door and asked for a small donation towards the local swimming pool.I gave him a glass of water.'])[0]
print 'We can see the probability with which the sentence is judged, sentence is non-joke : If you like Bill Bob in this film, you should also check out Dead Man'
print nb_detector.predict_proba(['If you like Bill Bob in this film, you should also check out Dead Man'])[0]

#print nb_detector.predict(['Hi Mom, How are you?'])[0]
#print nb_detector.predict(['WINNER! Credit for Free @ Lexington Corporation of 100000$!!'])[0]

print 'Predicting : Today a man knocked on my door and asked for a small donation towards the local swimming pool.I gave him a glass of water.'
print nb_detector.predict(['Today a man knocked on my door and asked for a small donation towards the local swimming pool.I gave him a glass of water.'])[0]

print 'Predicting : My review of the product is nothing but great.'
print nb_detector.predict(['My review of the product is nothing but great.'])[0]

predictions = nb_detector.predict(msg_test)
#print classification_report(label_test, predictions)

pipeline_svm = Pipeline([
    ('bow', CountVectorizer(analyzer=split_into_lemmas)),
    ('tfidf', TfidfTransformer()),
    ('classifier', SVC()),  
])

param_svm = [
  {'classifier__C': [1, 10, 100, 1000], 'classifier__kernel': ['linear']},
  {'classifier__C': [1, 10, 100, 1000], 'classifier__gamma': [0.001, 0.0001], 'classifier__kernel': ['rbf']},
]

grid_svm = GridSearchCV(
    pipeline_svm, 
    param_grid=param_svm,  
    refit=True,  
    n_jobs=-1,  
    scoring='accuracy',  
    cv=StratifiedKFold(label_train, n_folds=5),  
)

svm_detector = grid_svm.fit(msg_train, label_train)
#print svm_detector.grid_scores_

print 'Predicting : Today a man knocked on my door and asked for a small donation towards the local swimming pool.I gave him a glass of water.'
print svm_detector.predict(["Today a man knocked on my door and asked for a small donation towards the local swimming pool.I gave him a glass of water."])[0]
print 'Predicting : The second drawback doesnt apply to me but I read in one review that this doesnt support Mac.'
print svm_detector.predict(["The second drawback doesnt apply to me but I read in one review that this doesnt support Mac."])[0]


print(sys.version)