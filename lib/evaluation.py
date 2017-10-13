import matplotlib
matplotlib.use('Agg')
import os, json, numpy as np, math
from phase_interface import FileBasedProcess, ONE2ONE
import somoclu, json  
import matplotlib.pyplot as plt
from operator import itemgetter
from collections import defaultdict, namedtuple
from pprint import PrettyPrinter
from sklearn.preprocessing import normalize

class HeatMap(FileBasedProcess):

    def __init__(self):
        super(HeatMap, self).__init__(ONE2ONE)

    def output_prop(self):
        return {
            "description": "", 
            "ext": "png"
        }
        
    @FileBasedProcess.preparation
    def run(self, phase, **phase_ctx): 
        phase_input = phase_ctx.get("phase_input")[0]
        result_path = phase_ctx.get("result_paths")[0]
        data = json.load(open(phase_input))
        data = namedtuple('GenericDict', data.keys())(**data)
        ssm = np.matrix(data.ssm, dtype=float)

        fig = plt.figure(figsize=(10, 10))
        axs = fig.add_subplot(111)
        axs.grid(False)
        axs.set_title("users heatmap")
        row_ssm = ssm[:len(data.rows), :len(data.rows)]
        np.fill_diagonal(row_ssm, 0.0)
        # normalize to magnify results
        row_ssm = normalize(row_ssm, norm='l1', axis=1)
        axs.imshow(row_ssm, cmap='hot', interpolation='nearest')
        fig.savefig(result_path)


class TopKUser(FileBasedProcess):

    def __init__(self, k):
        super(TopKUser, self).__init__(ONE2ONE)
        self.__k = k

    def output_prop(self):
        return {
            "description": self.__k, 
            "ext": "json"
        }
        
    @FileBasedProcess.preparation
    def run(self, phase, **phase_ctx): 
        phase_input = phase_ctx.get("phase_input")[0]
        result_path = phase_ctx.get("result_paths")[0]
        data = json.load(open(phase_input))
        data = namedtuple('GenericDict', data.keys())(**data)
        ssm = np.array(data.ssm, dtype=float)

        row_ssm = ssm[:len(data.rows), :len(data.rows)]
        asc_idx_m = row_ssm.argsort(axis=1).tolist()


        trans = json.load(open(phase_ctx.get("modeling_input")[0], 'r'))
        u2f_m = np.array(trans['upper_right'])
        f2u_m = np.array(trans['down_left'])

        output = defaultdict(list)
        for i, sim_idx in enumerate(asc_idx_m):
            print "topk for %s" % data.rows[i]
            # k - 1 due to self similarity always the highest
            #topkuser = map(lambda idx: (data.rows[idx], row_ssm[i, idx]), [j for j in sim_idx[-self.__k-1:][::-1] if row_ssm[i, j] > 0.0])
            mastermind = data.rows[i]
            mastermind_upload = u2f_m[i, :]
            for j, accomplice, prob in map(lambda idx: (idx, data.rows[idx], row_ssm[i, idx]), [j for j in sim_idx[-self.__k-1:][::-1] if row_ssm[i, j] > 0.0]):
                accomplice_download =  f2u_m[:, j]
                total_download_count = np.sum(accomplice_download)
                accesses = []
                inter_vec = mastermind_upload*accomplice_download
                for access_idx in inter_vec.argsort()[::-1]:
                    if inter_vec[access_idx] == 0.0: break

                    accesses.append((data.cols[access_idx], accomplice_download[access_idx]/total_download_count))


                output[mastermind].append({
                    "name": accomplice,
                    "prob": prob,
                    "accesses": accesses
                })
                

            #relevance_users = map(lambda idx: data.rows[idx], [j for j in sim_idx if row_ssm[i, j] > 0.0])
            #r_length = len(relevance_users)
            #topkuser = relevance_users[-self.__k-1:]
            
            #f.write("%s: %s, (total: %d)\n" % (data.rows[i], topkuser, r_length))

        json.dump(output, open(result_path, 'w'))


class SOFMView(FileBasedProcess):

    def __init__(self, cols, rows):
        super(SOFMView, self).__init__(ONE2ONE)
        self.__cols = cols
        self.__rows = rows

    def output_prop(self):
        return {
            "description": [self.__cols, self.__rows], 
            "ext": "png"
        }
        
    @FileBasedProcess.preparation
    def run(self, phase, **phase_ctx): 
        curr_input = phase_ctx.get("phase_input")
        result_paths = phase_ctx.get("result_paths")
        filepath = curr_input[0]
        index = json.load(open(filepath, 'r'))
        matrix = index["matrix"]
        term = index["term"]
        doc = index["doc"]

        greys = plt.get_cmap("Greys")
        data = np.asarray(matrix)
        som = somoclu.Somoclu(self.__cols, self.__rows, data=data.T, initialization="pca", maptype="toroid", gridtype="hexagonal")
        som.train()
        som.view_umatrix(bestmatches=True, filename=result_paths[0], colormap=greys)


class KNNGraph(FileBasedProcess):

    def __init__(self, topics, word_feature=True, alpha=None, beta=None, uniq_file=1000, t=3):
        super(KNNGraph, self).__init__(ONE2ONE)
        self.__topics = topics
        self.__alpha = 1.0/topics if alpha is None else alpha
        self.__beta = 1.0/topics if beta is None else beta
        self.__word_feature = word_feature
        self.__uniq_file = uniq_file
        self.__t = t
        self.__lda_model_cache_base = '/home/safesync/lda_models'


    def output_prop(self): 
        return {
            "description": [self.__topics, self.__word_feature, self.__alpha, self.__beta, self.__uniq_file, self.__t],
            "ext": "model"
        }


    @FileBasedProcess.preparation
    def run(self, phase,  **phase_ctx):
        from sklearn.externals import joblib
        from sklearn.preprocessing import normalize
        from sklearn.decomposition import LatentDirichletAllocation
        from sklearn.cluster import KMeans

        filepath = phase_ctx.get("phase_input")[0]
        result_path = phase_ctx.get("result_paths")[0]
        print "loading index"
        vsm_index = joblib.load(filepath)
        feature_m = vsm_index["matrix"]
        vsm_doc_list = vsm_index["doc"]
        # term-by-doc matrix


        model_result_name = "_".join([self.md5pickle(filepath)] + map(str, [self.__topics, self.__alpha, self.__beta]))
        model_result_path = "%s/%s" % (self.__lda_model_cache_base, model_result_name)
        print "check lda model: %s" % model_result_path

        if not os.path.exists(model_result_path):
            print "prepare model" 
            lda = LatentDirichletAllocation(n_topics=self.__topics, 
                                            doc_topic_prior=self.__alpha, 
                                            topic_word_prior=self.__beta, 
                                            max_iter=5, 
                                            n_jobs=1, 
                                            learning_method='online', 
                                            learning_offset=50., 
                                            random_state=0)
            print "LDA model fitting"
            # X : numpy array of shape [n_samples, n_features]
            # doc-by-topic
            doc_topic_distr = lda.fit_transform(feature_m.T)
            # topic-by-word
            # topic_word_dist = lda.components_
            print "output result"
            model = {
                "fitted_model": lda,
                "doc_topic_dist": doc_topic_distr,
                "term": vsm_index["term"]
            }
            joblib.dump(model, model_result_path)
        else:
            model = joblib.load(model_result_path)
            lda = model["fitted_model"]
            doc_topic_distr = model["doc_topic_dist"]


        print "load user term index"
        print phase_ctx["modeling_input"]
        user_term_idx = json.load(open(phase_ctx["modeling_input"][0], 'r'))
        users = user_term_idx["user"]
        output = dict()


        for user, doc_access_dict in users.items():
            if user in user_black_list: 
                print "skip user:", user
                continue

            print user, len(doc_access_dict)
            if len(doc_access_dict) > self.__uniq_file:
                print "real_size to large: %d > %d" % (len(doc_access_dict), self.__uniq_file)
                continue
                
            # transform the topic-feature space to word-feature space
            doc_ids = sorted(doc_access_dict.keys())

            feature_list = [doc_topic_distr[vsm_doc_list.index(doc_id), :] for doc_id in doc_ids]
            
            # pairwise js divergence, symmetric matrix
            real_size = len(feature_list)
            if real_size > self.__uniq_file:
                print "real_size to large: %d > %d" % (real_size, self.__uniq_file)
                continue

                if self.__word_feature: continue

                print "%d, do kmeans" % (self.__uniq_file)
                # do KMEANS
                km = KMeans(n_clusters=self.__uniq_file, init="k-means++", max_iter=300, n_init=10, verbose=0)
                km.fit(np.matrix(feature_list))
                feature_list = km.cluster_centers_
                print "kmeans done."


            if self.__word_feature:
                feature_list = map(lambda f: np.ravel(normalize(np.dot(f.reshape(1, -1), lda.components_), norm='l1')), feature_list)
            
            size = len(feature_list)
            dist_m = np.identity(size)
            for i in range(size):
                # exclude the self-distance
                for j in range(i+1, size):
                    dist_m[i, j] = self.__JS_distance(feature_list[i], feature_list[j])

            # symmetrize matrix
            dist_m = np.maximum(dist_m, dist_m.T)
            
            # ODIN algo.
            k = int(math.ceil(float(real_size) / 4))
            knn_graph = self.__build_knn_graph(dist_m, k, doc_ids)
            # init 
            doc_id_indegree = dict([(doc_id, 0) for doc_id in doc_ids])
            for doc_id, top_k_dict in knn_graph.items():
                for in_doc_id in top_k_dict.keys():
                    doc_id_indegree[in_doc_id] += 1
                
            anomaly_doc_ids = [doc_id for doc_id, deg in doc_id_indegree.items() if deg <= self.__t]

            output[user] = {"k": k, 
                            "t": self.__t, 
                            "anomaly_doc_ids": anomaly_doc_ids,
                            "doc_id_indegree": doc_id_indegree
            }

        joblib.dump(output, result_path)

    def __build_knn_graph(self, dist_m, k, doc_ids):
        knn_graph = dict()
        adjust_k = k

        for i, doc_id in enumerate(doc_ids):
            # To consider multiple tie of zero score
            close_zero = np.where(np.isclose(np.zeros(dist_m[i, :].size), dist_m[i, :]) == True)[0]
            if close_zero.size > k:
                adjust_k = close_zero.size

            # dict of doc_id: distance
            knn_graph[doc_id] = dict([(doc_ids[j], dist_m[i, j]) for j in dist_m[i, :].argsort()[:adjust_k]])

        return knn_graph

    def __KL_divergence(self, p, q):
        # if q(i) is 0 then value is 0
        return np.dot(p, np.where(q!=0, np.log2(p/q), 0))

    def __JS_distance(self, p, q):
        # 0 <= value <= log2(2 ,2) = 1, applied on two distributions
        if np.all(np.isclose(p, q)):
            return 0.0

        factor = 0.5
        m = factor*(p+q)
        return factor*(self.__KL_divergence(p, m)+self.__KL_divergence(q, m))


class TopicDispersion(FileBasedProcess):

    def __init__(self, topics, dispersion_func, word_feature=True, alpha=None, beta=None, uniq_file=1000):
        super(TopicDispersion, self).__init__(ONE2ONE)
        self.__dispersion_func = dispersion_func
        self.__topics = topics
        self.__alpha = 1.0/topics if alpha is None else alpha
        self.__beta = 1.0/topics if beta is None else beta
        self.__word_feature = word_feature
        self.__uniq_file = uniq_file
        self.__lda_model_cache_base = '/home/safesync/lda_models'

    def output_prop(self): 
        return {
            "description": [self.__topics, self.__word_feature, self.__alpha, self.__beta, self.__uniq_file, self.__dispersion_func.__name__],
            "ext": "model"
        }


    @FileBasedProcess.preparation
    def run(self, phase,  **phase_ctx):
        from sklearn.externals import joblib
        from sklearn.preprocessing import normalize
        from sklearn.decomposition import LatentDirichletAllocation
        from sklearn.cluster import KMeans

        filepath = phase_ctx.get("phase_input")[0]
        result_path = phase_ctx.get("result_paths")[0]
        print "loading index"
        vsm_index = joblib.load(filepath)
        feature_m = vsm_index["matrix"]
        vsm_doc_list = vsm_index["doc"]
        # term-by-doc matrix


        model_result_name = "_".join([self.md5pickle(filepath)] + map(str, [self.__topics, self.__alpha, self.__beta]))
        model_result_path = "%s/%s" % (self.__lda_model_cache_base, model_result_name)
        print "check lda model: %s" % model_result_path

        if not os.path.exists(model_result_path):
            print "prepare model" 
            lda = LatentDirichletAllocation(n_topics=self.__topics, 
                                            doc_topic_prior=self.__alpha, 
                                            topic_word_prior=self.__beta, 
                                            max_iter=5, 
                                            n_jobs=1, 
                                            learning_method='online', 
                                            learning_offset=50., 
                                            random_state=0)
            print "LDA model fitting"
            # X : numpy array of shape [n_samples, n_features]
            # doc-by-topic
            doc_topic_distr = lda.fit_transform(feature_m.T)
            # topic-by-word
            # topic_word_dist = lda.components_
            print "output result"
            model = {
                "fitted_model": lda,
                "doc_topic_dist": doc_topic_distr,
                "term": vsm_index["term"]
            }
            joblib.dump(model, model_result_path)
        else:
            model = joblib.load(model_result_path)
            lda = model["fitted_model"]
            doc_topic_distr = model["doc_topic_dist"]


        print "load user term index"
        print phase_ctx["modeling_input"]
        user_term_idx = json.load(open(phase_ctx["modeling_input"][0], 'r'))
        users = user_term_idx["user"]
        user_deviation = defaultdict(float)


        for user, doc_access_dict in users.items():

            print user, len(doc_access_dict)
                
            # transform the topic-feature space to word-feature space
            doc_ids = sorted(doc_access_dict.keys())
            feature_list = [doc_topic_distr[vsm_doc_list.index(doc_id), :] for doc_id in doc_ids]
            
            
            # pairwise js divergence, symmetric matrix
            real_size = len(feature_list)
            if real_size > self.__uniq_file:
                if self.__word_feature: continue

                print "%d, do kmeans" % (self.__uniq_file)
                # do KMEANS
                km = KMeans(n_clusters=self.__uniq_file, init="k-means++", max_iter=300, n_init=10, verbose=0)
                km.fit(np.matrix(feature_list))
                feature_list = km.cluster_centers_
                print "kmeans done."

            if self.__word_feature:
                feature_list = map(lambda f: np.ravel(normalize(np.dot(f.reshape(1, -1), lda.components_), norm='l1')), feature_list)

            
                
            size = len(feature_list)
            distances = [] # will produce n*(n-1)/2 distances
            for i in range(size):
                # exclude the self-distance
                for j in range(i+1, size):
                    distances.append(self.__JS_distance(feature_list[i], feature_list[j]))

            
            radius = self.__dispersion_func(distances)
            # due to eliminate self-distance
            if np.isnan(radius):
                radius = 0.0

            user_deviation[user] = {"radius": radius, "uniq_doc": real_size}

        joblib.dump(user_deviation, result_path)
            

    def __deviation(self, distances):
        # only consider uniform distribution, so std is enough
        return np.std(np.array(distances))

    def __KL_divergence(self, p, q):
        # if q(i) is 0 then value is 0
        return np.dot(p, np.where(q!=0, np.log2(p/q), 0))

    def __JS_distance(self, p, q):
        # 0 <= value <= log2(2 ,2) = 1, applied on two distributions
        if np.all(np.isclose(p, q)):
            return 0.0

        factor = 0.5
        m = factor*(p+q)
        return factor*(self.__KL_divergence(p, m)+self.__KL_divergence(q, m))
    


class LDA(FileBasedProcess):
    def __init__(self, topics, alpha=None, beta=None):
        super(LDA, self).__init__(ONE2ONE)
        self.__topics = topics
        self.__alpha = 1.0/topics if alpha is None else alpha
        self.__beta = 1.0/topics if beta is None else beta

    def output_prop(self):
#        desc = ""
        return {
            "description": [self.__topics, self.__alpha, self.__beta],
            "ext": "model"
        }

    @FileBasedProcess.preparation
    def run(self, phase,  **phase_ctx):
        from sklearn.decomposition import LatentDirichletAllocation
        from sklearn.externals import joblib

        filepath = phase_ctx.get("phase_input")[0]
        result_path = phase_ctx.get("result_paths")[0]
        print "loading index"
        index = joblib.load(filepath)
        feature_m = index["matrix"]
        term = index["term"]
        doc = index["doc"]
        print type(feature_m)

        # term-by-doc matrix

        print "prepare model" 
        lda = LatentDirichletAllocation(n_topics=self.__topics, 
                                        doc_topic_prior=self.__alpha, 
                                        topic_word_prior=self.__beta, 
                                        max_iter=5, 
                                        n_jobs=1, 
                                        learning_method='online', 
                                        learning_offset=50., 
                                        random_state=0)

        print "LDA model fitting"
        # X : numpy array of shape [n_samples, n_features]
        # doc-by-topic
        doc_topic_dist = lda.fit_transform(feature_m.T)
        # topic-by-word
        # topic_word_dist = lda.components_
        print "output result"
        model = {
            "fitted_model": lda,
            "doc_topic_dist": doc_topic_dist,
            "term": term
        }
        joblib.dump(model, result_path)

    

class KMeansCluster(FileBasedProcess):

    def __init__(self, k, approx=None, show=None, normalize=True):
        super(KMeansCluster, self).__init__(ONE2ONE)
        self.__approx = approx
        self.__k = k
        self.__norm = "normalize" if normalize else ""
    
    def output_prop(self):
        approx_name  = self.__approx.get_desc() if self.__approx else None
        return {
            "description": [self.__k, approx_name,self.__norm],
            "ext": "png"
        }
        
    @FileBasedProcess.preparation
    def run(self, phase, **phase_ctx): 
        from sklearn.cluster import KMeans
        from collections import Counter
        from sklearn.metrics import silhouette_score
        from sklearn.externals import joblib

        filepath = phase_ctx.get("phase_input")[0]
        result_path = phase_ctx.get("result_paths")[0]
        print "loading index"
#        index = json.load(open(filepath, 'r'))
        index = joblib.load(filepath)
        feature_m = index["matrix"]
        term = index["term"]
        doc = index["doc"]

        # term-by-doc matrix
        #feature_m = np.matrix(matrix)
        if self.__approx:
            feature_m = self.__approx.process(feature_m)

        km = KMeans(n_clusters=self.__k, init="k-means++", max_iter=300, n_init=10, verbose=0)
        # transpose feature matrix for document clustering
        print "feature shape: %s" % str(feature_m.T.shape)
        km.fit(feature_m.T)
        print "silhouette score(cosine):", silhouette_score(feature_m.T, km.labels_, metric='cosine')
        print "silhouette score(euclidean):", silhouette_score(feature_m.T, km.labels_, metric='euclidean')
        # sort the cluster by number of documents in descending order
        label_list = sorted((Counter(km.labels_)).items(), key=itemgetter(1), reverse=True)
        # sort the centers value args in descending order
        cluster_arg_m = km.cluster_centers_.argsort()[:, ::-1]
        cluster_value_m = np.sort(km.cluster_centers_, axis=1)[:, ::-1]

        print "load user term index"
        print phase_ctx["modeling_input"]
        user_term_idx = json.load(open(phase_ctx["modeling_input"][0], 'r'))
        terms = user_term_idx["term"]
        docs = user_term_idx["doc"]
        users = user_term_idx["user"]
        
        topics = list()
        # TODO heuristic value
        tolerance_min = 9.5e-5
        for cluster_id, (label, number_of_docs) in enumerate(label_list):
            cv = cluster_value_m[label, :]
            cv = cv[cv > tolerance_min]
            if cv.size == 0: continue

            value_args = cluster_arg_m[label, :cv.size]
            value_terms = map(lambda idx: term[idx], value_args)
            # min scaling
            cv = cv * (1 / cv.min())
            pp = PrettyPrinter()
            # for word cloud
            term_weight = dict(zip(value_terms, cv))

            # for user usage
            user_cluster_access = defaultdict(int)
            doc_id_access = defaultdict(int)
            cluster_doc_ids = [doc[idx] for idx, tag in enumerate(km.labels_) if tag == label]
            for username, doc_access_dict in users.items():
                a, b = frozenset(doc_access_dict.keys()), frozenset(cluster_doc_ids)
                # misclassification error
                access_doc_count = map(lambda doc_id: doc_access_dict.get(doc_id, 0), a.intersection(b))
                if access_doc_count != []: 
                    max_count = max(access_doc_count)
                    count = float(sum(access_doc_count))
                    # to filter no use users
                    if count:
                        user_cluster_access[username] += count * (1+1e-10 - (max_count/count))

                for doc_id in a.intersection(b):
                    doc_id_access[doc_id] += doc_access_dict.get(doc_id, 0)


            # for important files  number of access
            # TODO nearest(C) * count(C)
            topics.append({
                "term_weight": term_weight,
                "num_of_term": cv.size,
                "num_of_doc": number_of_docs,
                "user_cluster_access": dict(user_cluster_access),
                "doc_id_access": dict(doc_id_access)
            })
            
        

        # save as word cloud
        from wordcloud import WordCloud
#        import seaborn as sb
        import linecache as lc
        w, h = 300, 200
        figsize = (8*2, 8.0*h/w*len(topics))
        
        wc = WordCloud(width=w, height=h)
        fig = plt.figure(figsize=figsize)
        total_ax = len(topics)*2
        for idx, topic in enumerate(topics):
            axs = fig.add_subplot(len(topics), 2, idx*2+1)
            axs.grid(False)
            axs.set_title("cluster %d, #docs=%d, #terms=%d" % (idx, topic["num_of_doc"], topic["num_of_term"]))
            img = wc.generate_from_frequencies(topic["term_weight"])
            axs.imshow(img, aspect="auto")
            pp.pprint(topic["user_cluster_access"])

            img = wc.generate_from_frequencies(topic["user_cluster_access"])
            axs = fig.add_subplot(len(topics), 2, idx*2+2)
            axs.grid(False)
            axs.set_title("#users=%d, #access=%d" % (len(topic["user_cluster_access"]), sum(topic["user_cluster_access"].values())))
            axs.imshow(img, aspect="auto")
            
            # construct heatmap matrix

        fig.savefig(result_path)
        plt.close('all')


class TopicTerms(FileBasedProcess):
    def run(self):
        topk_term = 3
        for topic in topic_list[:1]:
            topic_map = dict()

            topic_doc_set = set()
            for term in topic[:topk_term]:
                topic_doc_set.update(terms[term].keys())

            
            topic_doc_list = list(topic_doc_set)
            for doc_id in topic_doc_list[:4]:
                print doc_id, docs[doc_id]
                

            topic_map["docs"] = map(lambda doc_id: docs[doc_id], topic_doc_list)
            
            topic_map["users"] = defaultdict(int)
            for doc_id in topic_doc_set:
                for user in users:
                    if doc_id in users[user]:
                        # access count sum
                        topic_map["users"][user] += users[user][doc_id]

            output.append(topic_map)

        
            
        json.dump(output, open(target_path, 'w'))
        
