import os, numpy as np, json, math, sys
from sklearn.preprocessing import Normalizer
from phase_interface import FileBasedProcess, ONE2ONE
from operator import itemgetter
import numpy as np, codecs
from collections import namedtuple
from scipy.sparse import lil_matrix
from sklearn.preprocessing import normalize


class RWRQuery(FileBasedProcess):
    def __init__(self, querynode, c=0.15, eps=0.1):
        super(RWRQuery, self).__init__(ONE2ONE)
        self.__c = c
        self.__eps = eps
        self.__querynode = querynode

    def output_prop(self):
        return {
            "description": [self.__c, self.__eps, self.__querynode], 
            "ext": "json"
        }

    @FileBasedProcess.preparation
    def run(self, phase, **phase_ctx): 
        result_path = phase_ctx.get("result_paths")[0]
        index_file = phase_ctx.get("phase_input")[0]
        data = json.load(open(index_file))
        data = namedtuple('GenericDict', data.keys())(**data)
        upper_right = np.array(data.upper_right)
        down_left = np.array(data.down_left)

        self.__cut = upper_right.shape[0]
        
        print upper_right.shape
        print (upper_right[upper_right>0]).size / float(upper_right.size)
        print down_left.shape
        print (down_left[down_left>0]).size / float(down_left.size)
        print "shape %d, %d" % (len(data.rows), len(data.cols))

        # normalize row
        ijmtpm = normalize(upper_right, norm='l1', axis=1)
        jimtpm = normalize(down_left, norm='l1', axis=1)
        total_len = len(data.rows) + len(data.cols)

        try:
            row_idx = data.rows.index(self.__querynode)
        except ValueError:
            transform = {
                "ssm": 0,
                "query": self.__querynode,
                "files": None
            }
            with open(result_path, 'w') as dest:
                dest.write(json.dumps(transform))
                dest.flush()
            print "cannot find query node %s" % self.__querynode
            print "normality:0.0|(0,0)"
            return 


        ssm = np.matrix(np.zeros(total_len), dtype=float)
        iter_list, nz_list = [], []
        
        print row_idx
        q = np.zeros(total_len, dtype=float)
        q[row_idx] = 1.0
        u, iters, nz_ratio = self.__run_rwr(q, ijmtpm, jimtpm)
        ssm = np.append(ssm, np.matrix(u), axis=0)
        iter_list.append(iters)
        nz_list.append(nz_ratio)

        # remove the first row
        ssm = ssm[1:]
        iter_list = np.array(iter_list)
        nz_list = np.array(nz_list)
        print "iters: max:%d, min:%d, avg:%.3f, std:%.3f" % (iter_list.max(), iter_list.min(), np.average(iter_list), np.std(iter_list))
        print "non-zero ratio: max:%.3e, min:%.3e, avg:%.3e, std:%.3e" % (nz_list.max(), nz_list.min(), np.average(nz_list), np.std(nz_list))
        print "normality:%.5e|%s" % (np.sum(ssm), ssm.shape)

        transform = {
            "ssm": ssm.tolist(),
            "rows": data.rows,
            "cols": self.__querynode
        }

        with open(result_path, 'w') as dest:
            dest.write(json.dumps(transform))
            dest.flush()



    def __run_rwr(self, q, ijmtpm, jimtpm):
        cu = q
        iters, diff = 0, 0
        ijcn = normalize(ijmtpm, norm='l1', axis=0)
        jicn = normalize(jimtpm, norm='l1', axis=0)

        while True:
            right = np.dot(cu[:self.__cut], ijcn)
            left = np.dot(cu[self.__cut:], jicn)
            nu = (1-self.__c) * np.concatenate((left, right)) + self.__c * q
            # ord = 1, max(sum(abs(x), axis=0))
            diff = np.linalg.norm(nu-cu, ord=1) 
            cu = nu
            if diff < self.__eps: break   
            iters += 1
            
        nz_ratio = float((cu[cu > 0.0]).size) / cu.size
        return cu, iters, nz_ratio



#random walk with restart similarity matrix
class RWRSM(FileBasedProcess):
    def __init__(self, c=0.15, eps=0.1):
        super(RWRSM, self).__init__(ONE2ONE)
        self.__c = c
        self.__eps = eps

    def output_prop(self):
        return {
            "description": [self.__c, self.__eps], 
            "ext": "json"
        }

    @FileBasedProcess.preparation
    def run(self, phase, **phase_ctx): 
        result_path = phase_ctx.get("result_paths")[0]
        index_file = phase_ctx.get("phase_input")[0]
        data = json.load(open(index_file))
        data = namedtuple('GenericDict', data.keys())(**data)
        upper_right = np.matrix(data.upper_right)
        down_left = np.matrix(data.down_left)

        self.__cut = upper_right.shape[0]
        
        print upper_right.shape
        print (upper_right[upper_right>0]).size / float(upper_right.size)
        print down_left.shape
        print (down_left[down_left>0]).size / float(down_left.size)
        print "shape %d, %d" % (len(data.rows), len(data.cols))

        # normalize row
        ijmtpm = normalize(upper_right, norm='l1', axis=1)
        jimtpm = normalize(down_left, norm='l1', axis=1)
        total_len = len(data.rows) + len(data.cols)

        fa = np.sum(down_left > 0, axis=1)
        print "file to unique user: max=%d, min=%d, avg=%.3e, std=%.3e" % (fa.max(), fa.min(), np.average(fa), np.std(fa))
        fa = np.sum(upper_right > 0, axis=1)
        print "user to unique file: max=%d, min=%d, avg=%.3e, std=%.3e" % (fa.max(), fa.min(), np.average(fa), np.std(fa))

        # init a dimension space
        ssm = np.matrix(np.zeros(total_len), dtype=float)
        iter_list, nz_list = [], []
        
        #for i in range(1, total_len):
        for i in range(len(data.rows)):
            print "%d %d" % (i, total_len)
            q = np.zeros(total_len, dtype=float)
            q[i] = 1.0
            u, iters, nz_ratio = self.__run_rwr(q, ijmtpm, jimtpm)
            ssm = np.append(ssm, np.matrix(u), axis=0)
            iter_list.append(iters)
            nz_list.append(nz_ratio)


        # remove the first row
        ssm = ssm[1:]
        iter_list = np.array(iter_list)
        nz_list = np.array(nz_list)
        print "iters: max:%d, min:%d, avg:%.3f, std:%.3f" % (iter_list.max(), iter_list.min(), np.average(iter_list), np.std(iter_list))
        print "non-zero ratio: max:%.3e, min:%.3e, avg:%.3e, std:%.3e" % (nz_list.max(), nz_list.min(), np.average(nz_list), np.std(nz_list))

#        batch style
#        qm = np.identity(total_len)
#        ssm = self.__batch_rwr(qm, ijmtpm, jimtpm)

        transform = {
            "ssm": ssm.tolist(),
            "rows": data.rows,
            "cols": data.cols 
        }

        with open(result_path, 'w') as dest:
            dest.write(json.dumps(transform))
            dest.flush()
            

    def __batch_rwr(self, qm, ijmtpm, jimtpm):
        ijcn = normalize(ijmtpm, norm='l1', axis=0)
        jicn = normalize(jimtpm, norm='l1', axis=0)
        cu = qm

        idx = 0
        while True:
            right = np.dot(cu[:, :self.__cut], ijcn)
            left = np.dot(cu[:, self.__cut:], jicn)
            nu = (1-self.__c) * np.concatenate((left, right), axis=1) + self.__c * qm
            diff_arr = np.linalg.norm(nu-cu, axis=1, ord=1)
            print (diff_arr > self.__eps).size

            if reduce(lambda x,y: x and y, (diff_arr < self.__eps).tolist()):
                break

            idx += 1
            cu = nu

        return cu


    def __run_rwr(self, q, ijmtpm, jimtpm):
        cu = q
        iters, diff = 0, 0
        ijcn = normalize(ijmtpm, norm='l1', axis=0)
        jicn = normalize(jimtpm, norm='l1', axis=0)

        while True:
            right = np.dot(cu[:self.__cut], ijcn)
            left = np.dot(cu[self.__cut:], jicn)
            nu = (1-self.__c) * np.concatenate((left, right)) + self.__c * q
            # ord = 1, max(sum(abs(x), axis=0))
            diff = np.linalg.norm(nu-cu, ord=1) 
            cu = nu
            if diff < self.__eps: break   
            iters += 1
            
        nz_ratio = float((cu[cu > 0.0]).size) / cu.size
        return cu, iters, nz_ratio



class TfIdfMatrix(FileBasedProcess):

    def __init__(self, normalize=True, idf=True):
        super(TfIdfMatrix, self).__init__(ONE2ONE)
        self.__normalize = normalize
        self.__idf_option = idf
        

    def output_prop(self):
        return {
            "description": sorted(["matrix", "term", "doc", self.__normalize, self.__idf_option]), 
            "ext": "model"
        }

    @FileBasedProcess.preparation
    def run(self, phase, **phase_ctx): 
        from sklearn.externals import joblib
        from scipy.sparse import csr_matrix
        result_path = phase_ctx.get("result_paths")[0]
        index_file = phase_ctx.get("phase_input")[0]
        index = json.load(codecs.open(index_file, 'r', 'utf-8'))

        print "user, term, doc"
        print len(index["user"]), len(index["term"]), len(index["doc"])

        # TODO change modeling
        # vector space model
        terms = index["term"]
        docs = index["doc"]
        users = index["user"]
        number_of_terms = len(terms)
        number_of_docs = len(docs)

        # build a tf-idf matrix
        vsm = lil_matrix((number_of_terms, number_of_docs), dtype=float)

        # keep the order
        term_list = sorted(terms.keys())
        doc_list = sorted(docs.keys())
        doc_dict = dict([(doc, pos) for pos, doc in enumerate(doc_list)])

        weighted_func = lambda a, b: self.__tfidf(a, b, terms, number_of_docs)
        if not self.__idf_option:
            weighted_func = lambda a, b: self.__tf_count(a, b, terms)
            
        
        for i, term in enumerate(term_list):
            for j, doc in map(lambda doc_id: (doc_dict[doc_id], doc_id), terms[term].keys()):
                vsm[i, j] = weighted_func(term, doc)

        print "finished a tf-idf matrix"

        if self.__normalize:
            vsm = (Normalizer(norm='l2').fit_transform(vsm.T)).T

        print "output results"
        result = {
            # computing distance matrix
            "matrix": csr_matrix(vsm),
            "term": term_list,
            "doc": doc_list
        }
        joblib.dump(result, result_path)

    def __tf(self, term, doc, term_dict):
        if term in term_dict and doc in term_dict[term]:
            return len(term_dict[term][doc]) / float(len(term_dict))
            
        return 0.0

    def __tf_count(self, term, doc, term_dict):
        if term in term_dict and doc in term_dict[term]:
            return len(term_dict[term][doc])
            
        return 0.0

    def __idf(self, term, term_dict, number_of_docs):
        if term in term_dict:
            # plus 1.0 to avoid zero divided
            appear_docs = 1.0 if len(term_dict[term]) == 0 else float(len(term_dict[term]))
            return math.log(number_of_docs / appear_docs)

        return 0.0

    def __tfidf(self, term, doc, term_dict, number_of_docs):
        tf =  self.__tf(term, doc, term_dict) 
        if tf != 0.0:
            return tf * self.__idf(term, term_dict, number_of_docs)

        return 0.0



