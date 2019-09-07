from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener
import json

import sys
sys.path.append('/home/dandy/Keep/Natural-Language-Processing-With-Python-and-NLTK/Python Files/19. Sentiment Analysis Module')
import sentiment_mod as s
##import pickled_algos

sys.path.append('/home/dandy/Keep/Natural-Language-Processing-With-Python-and-NLTK-Videos')
# import twitterapistuff # this will not work use the below importing statement
from twitterapistuff import *

##twitterapistuff.py contains 
##01. consumer key
##02. consumer secret
##03. access token
##04. access secret

class listener(StreamListener):
        
    def on_data(self, data):
        try:
            all_data = json.loads(data)
            tweet = all_data["text"]
            sentiment_value, confidence = s.sentiment(tweet)
            print(tweet, sentiment_value, confidence)
            if confidence*100 >= 80:
                output = open("twitter-out.txt","a")
                output.write(sentiment_value)
                output.write('\n')
                output.close()
            return True
        except:
            return True

    def on_error(self, status):
        print(status)

auth = OAuthHandler(ckey, csecret)
auth.set_access_token(atoken, asecret)

twitterStream = Stream(auth, listener())
twitterStream.filter(track=["happy"])
