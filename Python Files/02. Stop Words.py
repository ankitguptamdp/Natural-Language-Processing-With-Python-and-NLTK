from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

example_sentence="This is an example showing off stop word filtration."

stop_words = set(stopwords.words("english"))
##print(stop_words)

words=word_tokenize(example_sentence)
##01
##filtered_sentence = []
##
##for w in words:
##    if w not in stop_words:
##        filtered_sentence.append(w)

##02
##filtered_sentence = [w for w in words if w not in stop_words]

##03
filtered_sentence = [w for w in words if not w in stop_words]

##All three are correct
print(filtered_sentence)
