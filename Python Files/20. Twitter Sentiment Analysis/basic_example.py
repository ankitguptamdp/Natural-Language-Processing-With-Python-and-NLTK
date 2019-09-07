from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener

import sys
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
        print(data)
        return(True)

    def on_error(self, status):
        print(status)

auth = OAuthHandler(ckey, csecret)
auth.set_access_token(atoken, asecret)

twitterStream = Stream(auth, listener())
twitterStream.filter(track=["car"])
