import nltk
import random
from nltk.corpus import movie_reviews
import pickle

from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
# sdc = stochastic gradient descent
# svm = support vector machines
# SVC = support vector classifier
# NuSVC - You can give the no of support vector

from nltk.classify import ClassifierI # interface
from statistics import mode

from nltk.tokenize import word_tokenize


class VoteClassifier(ClassifierI):
    
    def __init__(self,*classifiers):
        self._classifiers = classifiers

    def classify(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        return mode(votes)

    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)
        return conf

short_pos = open("../Texts/positive.txt","r").read()
short_neg = open("../Texts/negative.txt","r").read()

documents = []

for r in short_pos.split('\n'):
    documents.append((r,"pos"))

for r in short_neg.split('\n'):
    documents.append((r,"neg"))

all_words = []

short_pos_words = word_tokenize(short_pos)
short_neg_words = word_tokenize(short_neg)

for w in short_pos_words:
    all_words.append(w.lower())

for w in short_neg_words:
    all_words.append(w.lower())

all_words = nltk.FreqDist(all_words)

word_features = list(all_words.keys())[:5000]

def find_features(document):
    words = word_tokenize(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)
    return features

featuresets = [(find_features(rev), category) for (rev, category) in documents]

random.shuffle(featuresets)

# data example :
training_set = featuresets[:10000]
testing_set = featuresets[10000:]

# 01.
# Original Naive Bayes Classifier
classifier = nltk.NaiveBayesClassifier.train(training_set)
print("Original Naive Bayes Classifier Accuracy Percent : ",(nltk.classify.accuracy(classifier,testing_set))*100)
#classifier.show_most_informative_features(15)


# 02.
# Multinomial Naive Bayes Classifier
MultinomialNB_classifier = SklearnClassifier(MultinomialNB())
MultinomialNB_classifier.train(training_set)
print("MultinomialNB_classifier accuracy percent : ", (nltk.classify.accuracy(MultinomialNB_classifier, testing_set))*100)

# MultinomialNB_classifier.show_most_informative_features(15)
# AttributeError: 'SklearnClassifier' object has no attribute 'show_most_informative_features'
# This classifier doesn't have most_informative_features() function


# 03. 
# Bernoulli Naive Bayes Classifier
BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
BernoulliNB_classifier.train(training_set)
print("BernoulliNB_classifier accuracy percent : ", (nltk.classify.accuracy(BernoulliNB_classifier, testing_set))*100)


# 04.
# LogisticRegression Classifier
LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(training_set)
print("LogisticRegression_classifier accuracy percent : ", (nltk.classify.accuracy(LogisticRegression_classifier, testing_set))*100)


# 05.
# SGDClassifier
SGD_classifier = SklearnClassifier(SGDClassifier())
SGD_classifier.train(training_set)
print("SGD_classifier accuracy percent : ", (nltk.classify.accuracy(SGD_classifier, testing_set))*100)


# 06.
# LinearSVC
LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(training_set)
print("LinearSVC_classifier accuracy percent : ", (nltk.classify.accuracy(LinearSVC_classifier, testing_set))*100)


# 07.
# NuSVC
NuSVC_classifier = SklearnClassifier(NuSVC())
NuSVC_classifier.train(training_set)
print("NuSVC_classifier accuracy percent : ", (nltk.classify.accuracy(NuSVC_classifier, testing_set))*100)


# 08.
# VoteClassifier
voted_classifier = VoteClassifier(
                                  NuSVC_classifier,
                                  LinearSVC_classifier,                                  
                                  MultinomialNB_classifier,
                                  BernoulliNB_classifier,
                                  LogisticRegression_classifier)

print("voted_classifier accuracy percent : ",(nltk.classify.accuracy(voted_classifier, testing_set))*100)
