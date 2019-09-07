import nltk
import random
from nltk.corpus import movie_reviews
import pickle

documents = [(list(movie_reviews.words(fileid)),category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

random.shuffle(documents)

all_words = []

for w in movie_reviews.words():
    all_words.append(w.lower())

all_words = nltk.FreqDist(all_words)

# print(len(all_words)) # 39768

word_features = list(all_words.keys())[:3000]
# word_features = list(all_words.keys())[:30000]
# Naive Bayes Algorithm Accuracy Percent :  0
# And it is taking 2 minutes of time.
# For larger dataset it is failing very badly like 0 percent accuracy.


def find_features(document):
    words = set(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)
    return features

##print((find_features(movie_reviews.words('neg/cv000_29416.txt'))))

featuresets = [(find_features(rev), category) for (rev, category) in documents]

training_set = featuresets[:1900] # training_set = featuresets[:19000]
testing_set = featuresets[1900:] # testing_set = featuresets[19000:]

# posterior = prior_occurances * liklihood / evidence

# Trainig and testing
classifier = nltk.NaiveBayesClassifier.train(training_set)
print("Naive Bayes Algorithm Accuracy Percent : ",(nltk.classify.accuracy(classifier,testing_set))*100)
classifier.show_most_informative_features(15)

# Saving pickle
save_classifier = open("naivebayes.pickle","wb") # wb means write in bytes
pickle.dump(classifier,save_classifier)
save_classifier.close()

# Testing with pickle
classifier_file = open("naivebayes.pickle","rb") # rb means read in bytes
classifier2 = pickle.load(classifier_file)
classifier_file.close()
print("Naive Bayes Algorithm Accuracy Percent : ",(nltk.classify.accuracy(classifier2,testing_set))*100)
classifier2.show_most_informative_features(15)
