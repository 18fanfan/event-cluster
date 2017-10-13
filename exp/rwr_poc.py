import numpy as np, sys
from sklearn.preprocessing import normalize
from datetime import datetime


def rwr(title, c, q, ijmtpm, jimtpm, cut):
    print "%s, c=%.2f, shape=%s" % (title, c, str(ijmtpm.shape))
    tolerance = 0.1
    cu = q
    begin = datetime.now()
    idx, diff = 0, 0
    ijcn = normalize(ijmtpm, norm='l1', axis=0)
    jicn = normalize(jimtpm, norm='l1', axis=0)

    print ijcn
    print jicn

    while True:
        print cu
        sys.stderr.write("iter %d: diff=%.4f\n" % (idx, diff))
        right = np.dot(cu[:cut], ijcn)
        left = np.dot(cu[cut:], jicn)
        nu = (1-c) * np.concatenate((left, right)) + c * q
        diff = np.linalg.norm(nu-cu, ord=1) 
        if diff < tolerance: break   
        idx += 1
        cu = nu
        
    elapsed_time = datetime.now() - begin
    t = elapsed_time.seconds * 1000 + elapsed_time.microseconds / 1000.0
    print t

    nz = cu[cu > 0].size
    nz_ratio = float(nz) / cu.size
    print "non-zero ratio %.3f" % nz_ratio
    print np.nonzero(q*cu)
    print q, cu
    mask = np.bitwise_and(q == 1, cu != 0)
    print mask
    print cu[mask]
    print "cut max=%.4f, origin=%.4f" % (cu.max(), cu[mask])

    return idx, t, nz_ratio



num_of_users = 500
num_of_files = 30000
#iter 39: diff=0.1078 20s
c = 0.1
#c = 0.0
#c = 0.1
#iter 124: diff=0.1002, 62 s
#c = 0.05

#iter 2: diff=0.6100, 1s
# c = 0.9 

ij = np.matrix('1,1;1,0')
#ji = np.matrix('0,1;0,0')
ji = np.matrix('1,1;1,0')

# normalize row
ijmtpm = normalize(ij, norm='l1', axis=1)
jimtpm = normalize(ji, norm='l1', axis=1)

print ijmtpm
print jimtpm
for i in range(4):
    q = np.zeros(4, dtype=float)
    q[i] = 1.0
    rwr("test undirected graph""", c, q, ijmtpm, jimtpm, 2)


m_shape = (num_of_users, num_of_files)

# undirected graph
bi_adj_m = np.random.randint(1, 10000, m_shape)

# normalize row
mtpm = normalize(bi_adj_m, norm='l1', axis=1)
start_node = 0
q = np.zeros(num_of_users+num_of_files, dtype=float)
q[start_node] = 1.0
rwr("random undirected graph""", c, q, mtpm, mtpm.T, m_shape[0])


# directed graph
print "directed graph c=%.2f" % c
ij = np.random.randint(1, 10000, m_shape)
ji = (np.random.randint(1, 10000, m_shape)).T

# normalize row
ijmtpm = normalize(ij, norm='l1', axis=1)
jimtpm = normalize(ji, norm='l1', axis=1)
start_node = 0
q = np.zeros(num_of_users+num_of_files, dtype=float)
q[start_node] = 1.0
rwr("random directed graph""", c, q, ijmtpm, jimtpm, m_shape[0])


from scipy.sparse import rand
ratio = 0.001
ij = rand(*m_shape, density=ratio)
ji = (rand(*m_shape, density=ratio)).T

c = 0.05
# normalize row
ijmtpm = normalize(ij.todense(), norm='l1', axis=1)
jimtpm = normalize(ji.todense(), norm='l1', axis=1)
start_node = 0

time_list=[]
iter_list=[]
nonzero_list=[]
#for idx in range(m_shape[0] + m_shape[1]):
for idx in range(10):
    q = np.zeros(num_of_users+num_of_files, dtype=float)
    q[idx] = 1.0
    print idx
    it, t, nz = rwr("random directed graph""", c, q, ijmtpm, jimtpm, m_shape[0])
    iter_list.append(it)
    time_list.append(t)
    nonzero_list.append(nz)

ta = np.array(time_list)
ita = np.array(iter_list)
nza = np.array(nonzero_list)

print "total run %d" % ta.size
print "time: min=%.2f, max=%.2f, avg=%.2f, std=%.2f" % (ta.min(), ta.max(), np.average(ta), np.std(ta))
print "iter: min=%.2f, max=%.2f, avg=%.2f, std=%.2f" % (ita.min(), ita.max(), np.average(ita), np.std(ita))
print "non-zero ratio: min=%.2f, max=%.2f, avg=%.2f, std=%.2f" % (nza.min(), nza.max(), np.average(nza), np.std(nza))



