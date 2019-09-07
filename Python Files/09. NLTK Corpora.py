##import nltk
##
##print(nltk.__file__)
##
##/usr/lib/python3/dist-packages/nltk/__init__.py

##Inside data.py in the above directory

##if sys.platform.startswith('win'):
##    # Common locations on Windows:
##    path += [
##        str(r'C:\nltk_data'), str(r'D:\nltk_data'), str(r'E:\nltk_data'),
##        os.path.join(sys.prefix, str('nltk_data')),
##        os.path.join(sys.prefix, str('lib'), str('nltk_data')),
##        os.path.join(
##            os.environ.get(str('APPDATA'), str('C:\\')), str('nltk_data'))
##    ]
##else:
##    # Common locations on UNIX & OS X:
##    path += [
##        str('/usr/share/nltk_data'),
##        str('/usr/local/share/nltk_data'),
##        str('/usr/lib/nltk_data'),
##        str('/usr/local/lib/nltk_data'),
##        os.path.join(sys.prefix, str('nltk_data')),
##        os.path.join(sys.prefix, str('lib'), str('nltk_data'))
##    ]

from nltk.corpus import gutenberg
from nltk.tokenize import sent_tokenize

sample = gutenberg.raw("bible-kjv.txt")
tok = sent_tokenize(sample)

print(tok[1:3])
