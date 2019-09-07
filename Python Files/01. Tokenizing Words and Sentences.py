# import nltk

# nltk.download()

# tokenizing - word tokenizers and sentence tokenizers
# lexicon and corpora
# corpora is a body of text. ex: medical journals, presidential speeches, English Language
# lexicon is words and their meaning. ex:

# investor-speak and regular english-speak

# investor-speak 'bull' = someone who is positive about the market
# english-speak 'bull' = scary animal you don't want running at you

from nltk.tokenize import word_tokenize,sent_tokenize

example_text = "Hello Mr. Smith, how are you doing today? The weather is great and Python is awesome. The sky is pinkish-blue. You should not eat cardboard."

# print(sent_tokenize(example_text))

##print(word_tokenize(example_text))

##Ctrl+D for commenting and Ctrl+Shift+D for uncommenting

for i in word_tokenize(example_text):
    print(i)
