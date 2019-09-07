from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

##print(lemmatizer.lemmatize("cats"))
##print(lemmatizer.lemmatize("cacti"))
##print(lemmatizer.lemmatize("geese"))
##print(lemmatizer.lemmatize("rocks"))
##print(lemmatizer.lemmatize("python"))

##print(lemmatizer.lemmatize("better", pos="a")) # a means adjective
##
##print(lemmatizer.lemmatize("best", pos="a"))
##print(lemmatizer.lemmatize("run"))
##print(lemmatizer.lemmatize("run", pos="v")) # v means verb

print(lemmatizer.lemmatize("better")) # by default pos(Part of Speech) is set to noun ("n")


