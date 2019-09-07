import nltk
import random
from nltk.corpus import movie_reviews
import pickle

from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
# sdc = stochastic gradient descent
# svm = support vector machines
# SVC = support vector classifier
# NuSVC - You can give the no of support vector

from nltk.classify import ClassifierI # interface
from statistics import mode


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
        

documents = [(list(movie_reviews.words(fileid)),category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

##random.shuffle(documents)

all_words = []

for w in movie_reviews.words():
    all_words.append(w.lower())

all_words = nltk.FreqDist(all_words)

word_features = list(all_words.keys())[:3000]

def find_features(document):
    words = set(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)
    return features

featuresets = [(find_features(rev), category) for (rev, category) in documents]

### positive data example :
##training_set = featuresets[:1900]
##testing_set = featuresets[1900:]

# negative data example :
training_set = featuresets[100:]
testing_set = featuresets[:100]

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
### Gaussian Naive Bayes Classifier
##GaussianNB_classifier = SklearnClassifier(GaussianNB())
##GaussianNB_classifier.train(training_set)
##print("GaussianNB_classifier accuracy percent : ", (nltk.classify.accuracy(GaussianNB_classifier, testing_set))*100)

##TypeError: A sparse matrix was passed, but dense data is required. Use X.toarray() to convert to a dense numpy array.
# GaussianNB_classifier will give this error when using this type of training and testing set


# 04. 
# Bernoulli Naive Bayes Classifier
BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
BernoulliNB_classifier.train(training_set)
print("BernoulliNB_classifier accuracy percent : ", (nltk.classify.accuracy(BernoulliNB_classifier, testing_set))*100)

##from sklearn.linear_model import LogisticRegression, SGDClassifier
##from sklearn.svm import SVC, LinearSVC, NuSVC


# 05.
# LogisticRegression Classifier
LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(training_set)
print("LogisticRegression_classifier accuracy percent : ", (nltk.classify.accuracy(LogisticRegression_classifier, testing_set))*100)


# 06.
# SGDClassifier
SGD_classifier = SklearnClassifier(SGDClassifier())
SGD_classifier.train(training_set)
print("SGD_classifier accuracy percent : ", (nltk.classify.accuracy(SGD_classifier, testing_set))*100)


# 07.
# SVC
SVC_classifier = SklearnClassifier(SVC())
SVC_classifier.train(training_set)
print("SVC_classifier accuracy percent : ", (nltk.classify.accuracy(SVC_classifier, testing_set))*100)


# 08.
# LinearSVC
LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(training_set)
print("LinearSVC_classifier accuracy percent : ", (nltk.classify.accuracy(LinearSVC_classifier, testing_set))*100)


# 09.
# NuSVC
NuSVC_classifier = SklearnClassifier(NuSVC())
NuSVC_classifier.train(training_set)
print("NuSVC_classifier accuracy percent : ", (nltk.classify.accuracy(NuSVC_classifier, testing_set))*100)

voted_classifier = VoteClassifier(
                                  NuSVC_classifier,
                                  LinearSVC_classifier,                                  
                                  MultinomialNB_classifier,
                                  BernoulliNB_classifier,
                                  LogisticRegression_classifier)

print("voted_classifier accuracy percent : ",(nltk.classify.accuracy(voted_classifier, testing_set))*100)

print("Classification : ", voted_classifier.classify(testing_set[0][0]), "Confidence % : ", voted_classifier.confidence(testing_set[0][0])*100)
print("Classification : ", voted_classifier.classify(testing_set[1][0]), "Confidence % : ", voted_classifier.confidence(testing_set[1][0])*100)
print("Classification : ", voted_classifier.classify(testing_set[2][0]), "Confidence % : ", voted_classifier.confidence(testing_set[2][0])*100)
print("Classification : ", voted_classifier.classify(testing_set[3][0]), "Confidence % : ", voted_classifier.confidence(testing_set[3][0])*100)
print("Classification : ", voted_classifier.classify(testing_set[4][0]), "Confidence % : ", voted_classifier.confidence(testing_set[4][0])*100)
print("Classification : ", voted_classifier.classify(testing_set[5][0]), "Confidence % : ", voted_classifier.confidence(testing_set[5][0])*100)
