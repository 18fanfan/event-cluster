from lego_process import LegoProcess
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
import numpy as np


class KMean(LegoProcess):
    def __init__(self): pass
    def process(self, matrix):
        pass


class DbScan(LegoProcess):
    def __init__(self): pass
    def process(self, matrix):
        pass


class SOFM(LegoProcess):
    def __init__(self):pass
    def process(self, matrix):
        pass

# Latent semantic analysis aka Latent Semantic Indexing for IR
class RetainLSA(LegoProcess):
    def __init__(self, retain=100, normalize=True):
        #For LSA, a value of 100 is recommended.
        self.__retain = 100
        self.__normalize = normalize

    def process(self, matrix):
        svd = TruncatedSVD(n_components=self.__retain, algorithm="randomized", n_iter=5)
        doc_latent_m = svd.fit_transform(matrix)
        
        # the approximation matrix, I have tested with isclosed function
        approx_m = np.dot(doc_latent_m, svd.components_)
        if self.__normalize:
            return (Normalizer(norm='l2').fit_transform(approx_m))

        return approx_m

    def get_desc(self):
        return "%s_%s" % (self.__class__.__name__, str(self.__normalize))

class LeftRetainLSA(LegoProcess):
    def __init__(self, retain=100, normalize=True):
        #For LSA, a value of 100 is recommended.
        self.__retain = 100
        self.__normalize = normalize

    def process(self, matrix):
        svd = TruncatedSVD(n_components=self.__retain, algorithm="randomized", n_iter=5)
        doc_latent_m = svd.fit_transform(matrix)
        
        # the approximation matrix, I have tested with isclosed function
        if self.__normalize:
            return (Normalizer(norm='l2').fit_transform(svd.components_))

        return svd.components_

    def get_desc(self):
        return "%s_%s" % (self.__class__.__name__, str(self.__normalize))


class NMF(LegoProcess):
    def __init__(self):pass
    def process(self, matrix):
        pass

class SVDTopicTerms(LegoProcess):
    def __init__(self):pass
    def process(self, matrix):
        pass
       
def __matrix_approximation(self, m):
    print "top analysis"
    # TODO time consuming
    topic_num = 100
    U, sigma, VT = randomized_svd(m, n_components=topic_num, n_iter=5)
    print U.shape, sigma.shape, VT.shape
    ratio = sigma / sigma.sum() * 100

    # TODO threshold setup
    threshold = 1.4
    filter_n_topics = ratio[np.where(ratio > threshold)].size
    print ratio
    
    # TODO performance
    sigma_prime = np.concatenate([sigma[:filter_n_topics], np.zeros(topic_num - filter_n_topics)])
    sigma_prime = np.diag(sigma_prime)
    return np.dot(np.dot(U, sigma_prime), VT)


def __pairwise_cos_sim(self, m):
    # it will be a square matrix, doc length major
    # TODO remove print

    sim = defaultdict(float)
    for j in range(m.shape[1]):
        for k in range(j+1, m.shape[1]):
            sim[(j, k)] = self.__cosine_similarity(m[:, j], m[:, k])
            #print "\r%s" %  str((j, k)),
            print "%s, %.20f" % (str((j, k)), sim[(j,k)])

    print 
    sys.stdout.flush()
    # too slow
    # even applying sklm.pairwise.cosine_distances/sklm.pairwise.cosine_similarity
    #print "dot computing"
    #sim_m = np.dot(m.T, m)
    #_, number_of_docs = m.shape

    #print "norm computing"
    #for j in range(m.shape[1]):
    #    for k in range(m.shape[1]):
    #        if sim_m[j, k] != 0.0:
    #            # 2-norm
    #            sim_m[j, k] /= (np.linalg.norm(m[:, j]) * np.linalg.norm(m[:, k]))


def __semmetrize(self, m):
    return m + m.T - np.diag(m.diagonal())


def __cosine_similarity(self, v1, v2):
    # 2-norm
    # v1, v2 is numpy.array
    dot = np.dot(v1, v2.T)
    if dot != 0.0:
        return (dot / np.linalg.norm(v1) * np.linalg.norm(v2))

    return 0.0
