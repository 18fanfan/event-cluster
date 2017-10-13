from sklearn.neighbors import NearestNeighbors as NN
from sklearn.externals import joblib
import numpy as np, math, time

#User-defined distance:
#identifier  class name  args
#"pyfunc" PyFuncDistance  func
#Here func is a function which takes two one-dimensional numpy arrays, and returns a distance. 
#Note that in order to be used within the BallTree, the distance must be a true metric: i.e. it must satisfy the following properties
    #Non-negativity: d(x, y) >= 0
    #Identity: d(x, y) = 0 if and only if x == y
    #Symmetry: d(x, y) = d(y, x)
    #Triangle Inequality: d(x, y) + d(y, z) >= d(x, z)

def KL_divergence(p, q):
    # if q(i) is 0 then value is 0
    return np.dot(p, np.where(q!=0, np.log2(p/q), 0))

# 0 <= value <= log2(2 ,2) = 1, applied on two distributions
def JS_divergence(p, q):
    if np.all(np.isclose(p, q)):
        return 0.0

    factor = 0.5
    m = factor*(p+q)

    return factor*(KL_divergence(p, m)+KL_divergence(q, m))

def js_distance(p, q):
    return math.sqrt(JS_divergence(p, q))

def ball_tree(M):
    start = time.time()
    nbrs = NN(n_neighbors=k, algorithm='ball_tree', metric='pyfunc', metric_params={"func":js_distance}, leaf_size=100)
    nbrs.fit(M)
    print "ball tree elapsed time: ", time.time() - start
    return nbrs

def brute_force(M):
    total = 0
    feature_list = M
    size = feature_list.shape[0]
    dist_m = np.identity(size)
    for i in range(size)[:30]:
        # exclude the self-distance
        start = time.time()
        for j in range(i+1, size):
            dist_m[i, j] = js_distance(feature_list[i], feature_list[j])

        elapsed = time.time() - start
        print "brute force elapsed time: ", elapsed, i 
        total += elapsed

    # symmetrize matrix
    dist_m = np.maximum(dist_m, dist_m.T)
    print "avg: %.3f" % (total / 30.0)

#M = joblib.load('/tmp/balltree_data')
D = joblib.load('/home/safesync/lda_models/3757bb6d9586180087cb49f4730d5240_100_0.01_0.01')
M = D['doc_topic_dist'][np.random.choice(D['doc_topic_dist'].shape[0], 13000, replace=False), :]
k = int(math.ceil(M.shape[0] / 4.0))
print M.shape
print np.all(np.isclose(M.sum(axis=1), np.ones(M.shape[0])))


brute_force(M)













