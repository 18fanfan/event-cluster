# -*- coding: utf8 -*-
import numpy as np, sklearn, time, json, math, os
from sklearn.preprocessing import normalize
from sklearn.externals import joblib
from collections import defaultdict, Counter

def KL_divergence(p, q):
    # if q(i) is 0 then value is 0
    return np.dot(p, np.where(q!=0, np.log2(p/q), 0))

# 0 <= value <= log2(2 ,2) = 1, applied on two distributions
def JS_distance(p, q):
    if np.all(np.isclose(p, q)):
        return 0.0

    factor = 0.5
    m = factor*(p+q)

    return factor*(KL_divergence(p, m)+KL_divergence(q, m))


def odin(pair_dist, k, t):
    doc_id_score = defaultdict(int)
    for appear_id, dist_dict in pair_dist.items():
        # get top-k
        doc_id_score[appear_id]
        order_by_distance = sorted(dist_dict.items(), key=lambda x: x[1])

        # To consider multiple tie of zero score
        adjust_k = k
        close_zero = np.where(np.isclose(np.zeros(len(order_by_distance)), np.array([d for _, d in order_by_distance], dtype=float)) == True)[0]
        if close_zero.size > k:
            adjust_k = close_zero.size
            print "%s: multiple tie, after=%d, before=%d" % (appear_id, adjust_k, k)

        indegree_doc_ids = [doc_id for (doc_id, d) in (sorted(dist_dict.items(), key=lambda x: x[1])[:adjust_k])]
        for doc_id in indegree_doc_ids:
            doc_id_score[doc_id] += 1

    print doc_id_score
    stats = Counter([score for _, score in doc_id_score.items()])
    print sorted(stats.items(), key=lambda x: x[0])
    sorted_items = sorted(doc_id_score.items(), key=lambda x: x[1]) 
    t = sorted_items[0][1]
    print "lowest t: %d" % t
    t = t+1
    print "threashold t: %d, doc_id_score length: %d" % (t, len(doc_id_score))
    print [score for _, score in doc_id_score.items()]
    return [doc_id for doc_id, score in doc_id_score.items() if score <= t]
    
index = '/home/safesync/test/user_topic/data_transformation/7056dcd3bdf8a09c0fb8c59d129b7142_UserTermDocIndex_auth_rawrequest_2016-11-06_2017-01-07.json'
#index = '/home/safesync/test/user_topic/data_transformation/cebe9beadeb166f12ff0a37e2404e56b_UserTermDocIndex_auth_rawrequest_2016-11-06_2016-12-31.json'
#index = '/home/safesync/test/user_topic/data_transformation/d41c1eb81cde1ead32bd9dbf8558c2ad_UserTermDocIndex_auth_rawrequest_2016-11-06_2016-12-17.json'
#index = '/home/safesync/test/user_topic/data_transformation/030a1f7d1f1006c91b6bbe5f0ddc887c_UserTermDocIndex_auth_rawrequest_2016-11-06_2016-12-24.json'
#index = '/home/safesync/test/user_topic/data_transformation/38862fef9b783faf911c9d5deec9acb4_UserTermDocIndex_auth_rawrequest_2016-11-06_2016-12-10.json'

vsm = '/home/safesync/test/user_topic/modeling/d2cb50fe43ef8b9a452fea9cec9d896a_TfIdfMatrix_False_False_doc_matrix_term_2016-11-06_2017-01-07.model'
#vsm = '/home/safesync/test/user_topic/modeling/a45d89f5a60bf723c8f88e96f85150b8_TfIdfMatrix_False_False_doc_matrix_term_2016-11-06_2016-12-31.model'
#vsm = '/home/safesync/test/user_topic/modeling/49355cbb97eca5452b3d9aa3ef5e0f0b_TfIdfMatrix_False_False_doc_matrix_term_2016-11-06_2016-12-17.model'
#vsm = '/home/safesync/test/user_topic/modeling/d51dded2c26fb758ae340243474244a8_TfIdfMatrix_False_False_doc_matrix_term_2016-11-06_2016-12-24.model'
#vsm = '/home/safesync/test/user_topic/modeling/b1daa9fd04849f60364f297dd668f189_TfIdfMatrix_False_False_doc_matrix_term_2016-11-06_2016-12-10.model'


model = '/home/safesync/lda_models/abb7b7d774c003b0493e251bc7fe5e95_100_0.01_0.01'
model = '/home/safesync/lda_models/abb7b7d774c003b0493e251bc7fe5e95_100_0.1_0.01'
#model = '/home/safesync/lda_models/d4019aba5699182d6bb6305c6f032374_100_0.01_0.01'
#model = '/home/safesync/lda_models/8ae205c52dd8af5797e11fb12fbd9f7b_100_0.01_0.01'
#model = '/home/safesync/lda_models/52dbf3ab3462aabae9d4a6a4abfcd1da_100_0.01_0.01'
#model = '/home/safesync/lda_models/8b39d4affc6738632ae2d29726cb9882_100_0.01_0.01'
print "date: %s" % vsm

index = json.load(open(index, 'r'))


vsm = joblib.load(vsm)
feature_m = vsm["matrix"]
terms = vsm["term"]
doc = vsm["doc"]

#name = 'ext_sherwinc'
#name = 'ext_howardk'
#name = 'adah'
#name = 'tao-sheng_chen'
#name = 'carina_liao'
#name = 'szuyueh_fan'
#name = 'deon_chiu'
#name = 'edward_feng'
#name = 'morel_chang'
#name = 'mohamed_almuzanen'
name = 'koonleng_ng'

doc_idx2id = dict([(doc.index(k), k) for k in index['user'][name].keys()])

model = joblib.load(model)
#weeks = [joblib.load("%s/%s" % (read_base, filename)) for filename in sorted(os.listdir(read_base), key=lambda x: x.split('_')[-1])]
lda = model["fitted_model"]
doc_topic_distr = lda.transform(feature_m.T)

start = time.time()
pair_dist = dict()
for num, (idx, doc_id) in enumerate(doc_idx2id.items()):
    result = dict()
    sample = doc_topic_distr[idx, :]
    
    for j, ref_doc in doc_idx2id.items():
        if idx == j: continue
        result[ref_doc] = JS_distance(sample, doc_topic_distr[j])

    print "%d:%s %s" % (num+1, doc_id, repr(",".join(index["doc"][doc_id]["term_list"]).encode('utf-8')))
    for rank, (ref_doc, d) in enumerate(sorted(result.items(), key=lambda x: x[1])):
        print "%d, [%s], %.5f" % (rank+1, repr(" ".join(index["doc"][ref_doc]["term_list"]).encode('utf-8')), d)
    print
    pair_dist[doc_id] = result
    
    #for rank, (ref_doc, d) in enumerate(sorted(result.items(), key=lambda x: x[1])):
    #    print "%d, [%s], %.5f" % (rank+1, " ".join(map(lambda x: x.encode('utf-8'), index["doc"][ref_doc]["term_list"])), d)


print 
#for an_doc_id in odin(pair_dist, 10, 1):
#k = int(len(doc_idx2id) / 4.0)
k = int((len(doc_idx2id) / 4.0)) + 1
#k = int((len(doc_idx2id) / 3.0))

print "k=%d" % k 
anomalies = odin(pair_dist, k, 2)

for an_doc_id in anomalies:
    #print ",".join(map(lambda x: x.encode('utf-8'), index["doc"][an_doc_id]["term_list"]))
    print repr(",".join(index["doc"][an_doc_id]["term_list"]).encode('utf-8'))

    

print time.time() - start

