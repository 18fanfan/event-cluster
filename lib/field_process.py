# -*- coding: utf-8 -*-
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from urlparse import urlparse
from urllib import unquote
from lego_process import LegoProcess
from os.path import basename, splitext
import re


# apply the post-order procedure
# Leaf classes
class List2Scalar(LegoProcess):
    def __init__(self): pass

    def process(self, value):
        if type(value) is list:
            return value[0]

        return None
        

# Intermediate classes
class MethodAccessTransform(LegoProcess):
    def __init__(self, process_obj = None):
        super(MethodAccessTransform, self).__init__(process_obj)
        self.__mapping = {
            "put": "up",
            "filepatch": "up",
            "post": "up",
            "get": "down",
            "delta": "down"
        }

    def process(self, value):
        list_value = self.next(value)
        if list_value is None: return None

        method, uri = list_value

#        if access_path.startswith('/api/v1/versions/'):
#            access_path = '/'.join(access_path.split('/')[:-1])

        access_path = uri.path
        access_path = access_path.strip()
        
        if access_path.startswith('/api/v1/versions'):
            access_path = '/'.join(access_path.split('/')[:-1])

        if access_path.startswith('/api/v2/preview/'):
            access_path = '/'.join(access_path.split('/')[5:])
        if access_path.startswith('/api/v'):
            access_path = '/'.join(access_path.split('/')[4:])
        elif access_path.startswith('/dav/'):
            access_path = '/'.join(access_path.split('/')[2:])

        trans_access = access_path.lower()


        if trans_access == "" or trans_access[-1] == "/": 
            trans_access = None

        trans_method = self.__mapping.get(method.lower(), None)

        if trans_access and trans_method:

            return [trans_method, trans_access]


        return None


class ColumnFilter(LegoProcess):
    def __init__(self, filter_range, process_obj=None):
        super(ColumnFilter, self).__init__(process_obj)
        self.__slice = slice(*filter_range)

    def process(self, value):
        list_value = self.next(value)
        if list_value is None: return None
        return list_value[self.__slice]


class RawRequestFilter(LegoProcess):
    def __init__(self,process_obj = None):
        super(RawRequestFilter, self).__init__(process_obj)

    def process(self, value):
        request = self.next(value)
        if request is None: return None

        request = request.strip('"')
        method, uri, scheme = request.split()
        method = method.lower()
        uri = unquote(str(uri))
        uri = urlparse(uri)

        if uri.path.startswith('/api/v1/versions'):
            pass
            
        if method == 'get':
            if uri.path.startswith('/dav/'):
                pass
            elif re.match(r"/api/v1/[a-zA-Z]{4,5}/", uri.path):
                if "perPage" in uri.query:
                    uri = None
                else:
                    pass
            elif re.match(r"/api/v2/[a-zA-Z]{3,5}/", uri.path):
                if uri.path.startswith('/api/v2/url/') or \
                    uri.path.startswith('/api/v2/multi') or \
                    uri.path.startswith('/api/v2/user/'):
                    uri = None
                else:
                    pass
            elif uri.path.startswith('/api/v2/preview/'):
                pass
            else:
                uri = None
            
        if uri:
            return [method, uri, scheme]

        return None


class CleanAccessPath(LegoProcess):
    def __init__(self,process_obj = None):
        super(CleanAccessPath, self).__init__(process_obj)

    def process(self, value):
        list_value = self.next(value)
        if list_value is None: return None

        url = list_value[0]
        access_path = url.path
        access_path = access_path.strip()
        
        if access_path.startswith('/api/v1/versions'):
            access_path = '/'.join(access_path.split('/')[:-1])

        if access_path.startswith('/api/v2/preview/'):
            access_path = '/'.join(access_path.split('/')[5:])
        if access_path.startswith('/api/v'):
            access_path = '/'.join(access_path.split('/')[4:])
        elif access_path.startswith('/dav/'):
            access_path = '/'.join(access_path.split('/')[2:])

        if access_path == "" or access_path[-1] == "/": return None
        return access_path
    


class Tokenizer(LegoProcess):
    def __init__(self, process_obj = None, minlen = 1):
        super(Tokenizer, self).__init__(process_obj)
        self.__minlen = minlen
    
    def process(self, value):
        sentence = self.next(value)
        if sentence is None: return None
        tokenizer = RegexpTokenizer(r'[a-zA-Z0-9\x80-\xFF]{%d,}' % self.__minlen)
        words = tokenizer.tokenize(sentence)

        sep_uni_asc = ""
        for word in words:
            sep_uni_asc += word[0]
            for i in range(1, len(word)):
                if (ord(word[i]) - 127) * (ord(word[i-1]) - 127) < 0:
                    sep_uni_asc += ",%s" % word[i]
                else:
                    sep_uni_asc += word[i]

            sep_uni_asc += ","
            
        words = sep_uni_asc.split(',')[:-1]
        # to lower case
        return map(lambda w: str.lower(w), words)


class RemoveStopWord(LegoProcess):
    def __init__(self, process_obj = None):
        super(RemoveStopWord, self).__init__(process_obj)
    
    def process(self, value):
        word_list = self.next(value)
        if word_list is None: return None 
            
        stop_words = set(stopwords.words('english'))
        return filter(lambda w: not w in stop_words, word_list)
    
        
#TODO stemming
class Stemming(object):
    def __init__(self, process_obj):
        self.__process_obj = process_obj        

    def process(self, value): raise NotImplementedError


#TODO lemmatization
class Lemmatization(object):
    def __init__(self, process_obj):
        self.__process_obj = process_obj        

    def process(self, value): raise NotImplementedError


