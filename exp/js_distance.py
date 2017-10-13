import numpy as np, sklearn, time, json
from sklearn.preprocessing import normalize
from sklearn.externals import joblib

def KL_divergence(p, q):
    # if q(i) is 0 then value is 0
    return np.dot(p, np.where(q!=0, np.log2(p/q), 0.0))

# 0 <= value <= log2(2 ,2) = 1, applied on two distributions
def JS_distance(p, q):
    if np.all(np.isclose(p, q)):
        return 0.0

    factor = 0.5
    m = factor*(p+q)
    return factor*(KL_divergence(p, m)+KL_divergence(q, m))

a = np.array([0.1, 0.3])
b = np.array([0.2, 0.5])
c = np.array([0.7, 0.9])
print a, b, c
ab = JS_distance(a, b)
bc = JS_distance(b, c)
ac = JS_distance(a, c)
print ab, bc, ac, (ab+bc > ac)
print ab**0.5, bc**0.5, ac**0.5, ((ab**0.5)+(bc**0.5) > (ac**0.5))


exit(1)

    
index = '/home/safesync/test/user_topic/data_transformation/7056dcd3bdf8a09c0fb8c59d129b7142_UserTermDocIndex_auth_rawrequest_2016-11-06_2017-01-07.json'
index = '/home/safesync/test/user_topic/data_transformation/cebe9beadeb166f12ff0a37e2404e56b_UserTermDocIndex_auth_rawrequest_2016-11-06_2016-12-31.json'
index = '/home/safesync/test/user_topic/data_transformation/d41c1eb81cde1ead32bd9dbf8558c2ad_UserTermDocIndex_auth_rawrequest_2016-11-06_2016-12-17.json'
index = '/home/safesync/test/user_topic/data_transformation/38862fef9b783faf911c9d5deec9acb4_UserTermDocIndex_auth_rawrequest_2016-11-06_2016-12-10.json'

vsm = '/home/safesync/test/user_topic/modeling/d2cb50fe43ef8b9a452fea9cec9d896a_TfIdfMatrix_False_False_doc_matrix_term_2016-11-06_2017-01-07.model'
vsm = '/home/safesync/test/user_topic/modeling/a45d89f5a60bf723c8f88e96f85150b8_TfIdfMatrix_False_False_doc_matrix_term_2016-11-06_2016-12-31.model'
vsm = '/home/safesync/test/user_topic/modeling/49355cbb97eca5452b3d9aa3ef5e0f0b_TfIdfMatrix_False_False_doc_matrix_term_2016-11-06_2016-12-17.model'
vsm = '/home/safesync/test/user_topic/modeling/b1daa9fd04849f60364f297dd668f189_TfIdfMatrix_False_False_doc_matrix_term_2016-11-06_2016-12-10.model'


model = '/home/safesync/lda_models/abb7b7d774c003b0493e251bc7fe5e95_100_0.01_0.01'
model = '/home/safesync/lda_models/d4019aba5699182d6bb6305c6f032374_100_0.01_0.01'
model = '/home/safesync/lda_models/8ae205c52dd8af5797e11fb12fbd9f7b_100_0.01_0.01'
model = '/home/safesync/lda_models/8b39d4affc6738632ae2d29726cb9882_100_0.01_0.01'
print "date: %s" % vsm

index = json.load(open(index, 'r'))


vsm = joblib.load(vsm)
feature_m = vsm["matrix"]
terms = vsm["term"]
doc = vsm["doc"]

name = 'ext_sherwinc'
name = 'ext_howardk'
name = 'deon_chiu'
doc_idx2id = dict([(doc.index(k), k) for k in index['user'][name].keys()])

model = joblib.load(model)
lda = model["fitted_model"]
doc_topic_distr = lda.transform(feature_m.T)

start = time.time()
#for idx in np.random.choice(doc_topic_distr.shape[0], 1, replace=False):
for idx, doc_id in doc_idx2id.items():
#    a = idx
    result = dict()
    sample = doc_topic_distr[idx, :]
#    sample = np.ravel(normalize(np.dot(sample, lda.components_), norm='l1'))
    
    #for j in range(doc_topic_distr.shape[0]):
    for j, ref_doc in doc_idx2id.items():
#    for j in range(100):
        #ref = np.ravel(normalize(np.dot(doc_topic_distr[j], lda.components_), norm='l1'))
        ref = doc_topic_distr[j]
#        ref = np.ravel(normalize(np.dot(doc_topic_distr[j], lda.components_), norm='l1'))
        #print j, doc_topic_distr.shape[0], ref.shape, sample.shape
        result[ref_doc] = JS_distance(sample, ref)
        #result.append((j, JS_distance(sample, ref), doc_terms))

    print "title: " + ",".join(index["doc"][doc_id]["term_list"])
    for rank, (ref_doc, d) in enumerate(sorted(result.items(), key=lambda x: x[1])):
        print "%d, [%s], %.5f" % (rank+1, " ".join(map(lambda x: x.encode('utf-8'), index["doc"][ref_doc]["term_list"])), d)

    print

print time.time() - start

