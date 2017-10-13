import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np, os
from collections import defaultdict, namedtuple
from sklearn.externals import joblib
import json , pprint
from lib.event_analysis import EventAnalysis
from lib.anomaly_user import AnomalyUserDetection as AUD
from scipy.sparse import csr_matrix
from datetime import datetime, timedelta

pp = pprint.PrettyPrinter()


def dcg(rank_array):
    gain = rank_array
    discounts = np.log2(np.arange(rank_array.size) + 2)
    return np.sum(gain / discounts)

def exp_dcg(rank_array):
    gain = 2 ** rank_array - 1
    discounts = np.log2(np.arange(rank_array.size) + 2)
    return np.sum(gain / discounts)


def lb_ndcg(rank_array, dcg_func=dcg):
    # the list contain relavent score
    # we are not going to use 2**rel score duo to the score is smaller than 1
    cdcg = dcg_func(rank_array)
    # idcg
    upper_bound = dcg_func(np.sort(rank_array)[::-1])
    lower_bound = dcg_func(np.sort(rank_array))
    if upper_bound == lower_bound: return 1.0
    return (cdcg-lower_bound) / (upper_bound-lower_bound)


def jaccard_distance(dict1, dict2):
    # intersection jaccard seems better
    union_keys = frozenset(dict1.keys() + dict2.keys())
    inter = set(dict1.keys()).intersection(set(dict2.keys()))
    return 1 - (len(inter) / float(len(union_keys)))

def appear_distance(curr_dict, prev_dict):
    union_keys = frozenset(curr_dict.keys() + prev_dict.keys())
    diff = set(curr_dict.keys()) - set(prev_dict.keys())
    #print union_keys
    if len(union_keys) == 0: 
        return 0.0

    return len(diff) / float(len(union_keys))
    

def data_analysis(user_topic_base, user_collaboration_base):

    for i in range(1, 11)[::-1]:
        ea = EventAnalysis(2016, 11, 6, i*7-1, force_start=None, output_base=user_topic_base)
        ea.exp15_knn_graph()

    # User-Collaboration Analysis
    start = datetime(2016, 11, 6)
    for w in range(1, 10)[::-1]:
        c = start + timedelta(days=7*w)
        aud = AUD(c.year, c.month, c.day, unit=7, beta=3, wsize=0, data_start=start, force_start=None, output_base=user_collaboration_base)
        aud.exp2_directed_topk()

def make_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)

    return path

def topic_dispersion(weeks):
    topic_agg = defaultdict(dict)
    for idx, w in enumerate(weeks):
        for username, prop in w.items():
            if username in topic_agg:
                continue

            values = map(lambda x: (x[username]['radius'], x[username]['uniq_doc']), weeks[idx:])
            radius_values, uniq_doc_values = zip(*values)
            prefix_pad = [None]*idx
            topic_agg[username]['radius'] = prefix_pad + list(radius_values)
            topic_agg[username]['uniq_doc'] = prefix_pad + list(uniq_doc_values)

    return topic_agg


def topic_anomaly_access(weeks):
    topic_agg = defaultdict(dict)
    for idx, w in enumerate(weeks):
        for username, prop in w.items():
            if username in topic_agg:
                continue

            if username == '-': continue

            try: 
                values = map(lambda x: (x[username]['anomaly_doc_ids'], x[username]['doc_id_indegree']), weeks[idx:])
                anomaly_doc_ids , doc_id_indegree = zip(*values)
                prefix_pad = [None]*idx
                topic_agg[username]['anomaly_doc_ids'] = prefix_pad + list(anomaly_doc_ids)
                topic_agg[username]['doc_id_indegree'] = prefix_pad + list(doc_id_indegree)
            except KeyError as e:
                print "%s: %s" % (username, str(e))
                continue

    return topic_agg



base = '/home/safesync/tbox_3_to_5'
base = '/home/safesync/test'
make_folder(base)
user_topic_base = make_folder("%s/user_topic" % base)
user_collaboration_base = make_folder("%s/user_collaboration" % base)
figure_output =  make_folder("%s/figure_output" % base)
data_analysis(user_topic_base, user_collaboration_base)

# data aggregation
# aggregate topic analysis result
print "aggregate topic analyasis result"
read_base = '%s/evaluation' % user_topic_base

#print sorted(os.listdir(read_base), key=lambda x: x.split('_')[-1])
weeks = [joblib.load("%s/%s" % (read_base, filename)) for filename in sorted(os.listdir(read_base), key=lambda x: x.split('_')[-1])]
total_weeks = len(weeks)
topic_agg = topic_anomaly_access(weeks)


# aggregate user collaboration result
print "aggregate user collaboration analyasis result"
read_base = '%s/modeling' % user_collaboration_base
rs_list = ["%s/%s" % (read_base, filename) for filename in sorted(os.listdir(read_base), key=lambda x: x.split('_')[-1])]
user_agg = defaultdict(lambda: defaultdict(list))
user_rs_tol = 4e-5
damping_factor = 0.2

for i in range(1, len(rs_list)):
    print rs_list[i]
    prev = json.load(open(rs_list[i-1]))
    curr = json.load(open(rs_list[i])) 
    prev = namedtuple('GenericDict', prev.keys())(**prev)
    curr = namedtuple('GenericDict', curr.keys())(**curr)
    prev_ssm = np.array(prev.ssm, dtype=float)[:len(prev.rows), :len(prev.rows)]
    curr_ssm = np.array(curr.ssm, dtype=float)[:len(curr.rows), :len(curr.rows)]

    for r in range(len(curr.rows)):
        curr_username = curr.rows[r]
        try:
            prev_idx = prev.rows.index(curr_username)
            desc_idx = np.argsort(prev_ssm[prev_idx, :])[::-1]
            # name2score exclude self similarity
            prev_name2score = defaultdict(float, dict([(prev.rows[i], prev_ssm[prev_idx, i]) for i in desc_idx[1:] if prev_ssm[prev_idx, i] > user_rs_tol]))
            prev_name2rank = dict([(prev.rows[i], rank) for rank, i in enumerate(desc_idx) if prev_ssm[prev_idx, i] > user_rs_tol])
            max_value = max(prev_name2rank.values())
            # transform relevance score to rank order by desc
            prev_name2rank= dict([(k, max_value-v+1) for k, v in prev_name2rank.items()])

        except ValueError as e:
            user_agg[curr_username]["distance"].append(None)
            user_agg[curr_username]["filtered_score"].append(None)
            user_agg[curr_username]["filtered_users"].append(None)
            continue

        curr_desc_idx = np.argsort(curr_ssm[r, :])[::-1]
        # name2score exclude self similarity
        curr_name2score = defaultdict(float, dict([(curr.rows[i], curr_ssm[r, i]) for i in curr_desc_idx[1:] if curr_ssm[r, i] > user_rs_tol]))
        curr_rank = np.array([(prev_name2rank.get(curr.rows[i], 0.0), curr.rows[i]) for i in curr_desc_idx if curr_ssm[r, i] > user_rs_tol])
        curr_rank_array, curr_rank_name = zip(*curr_rank)
        curr_rank_array = np.array(map(float, curr_rank_array))

        # penalize the anonymouns user

        # 1-lb_ndcg to keep to 0.0
       # distance = 1-lb_ndcg(curr_rank_array)
#        if len(curr_name2score) > len(prev_name2score):
           # To solve the tail chaining issue
        distance = (1-lb_ndcg(curr_rank_array))*(1-damping_factor) + appear_distance(curr_name2score, prev_name2score)*damping_factor
        

        user_agg[curr_username]["distance"].append(distance)
        user_agg[curr_username]["filtered_score"].append(curr_name2score)
        user_agg[curr_username]["filtered_users"].append(curr_rank_name)


for username in user_agg.keys():
    size = len(user_agg[username]["distance"])
    user_agg[username]["distance"] = ([None]*(total_weeks-size))+user_agg[username]["distance"]
    
# prioritization
def prioritization(topic_agg, user_agg):
    change = 0.05
    priority = defaultdict(int)
    for username, topic_result in topic_agg.items():
        priority_score = 0
        user_result = user_agg[username]
        for i in range(1, len(topic_result["radius"])):
            test_score = 0
            if topic_result["radius"][i] is None or topic_result["radius"][i-1] is None: continue

            if topic_result["radius"][i] - topic_result["radius"][i-1] > change:
                test_score += 4
            elif topic_result["radius"][i] - topic_result["radius"][i-1] > 0.0:
                test_score += 2

            priority_score = max(priority_score, test_score)
            
            if user_result["distance"] == [] or user_result["distance"][i] is None or user_result["distance"][i-1] is None: continue
            
            if user_result["distance"][i] - user_result["distance"][i-1] > 0.0:
                test_score += 1

            priority_score = max(priority_score, test_score)


        priority[username] = priority_score

    return priority

print "prioritization"        
#priority = prioritization(topic_agg, user_agg)

priority = dict([(username, 1) for username in topic_agg.keys()])

# plot result
#for username in topic_agg.keys():
for username in topic_agg.keys():
    try:
        #idx = total_weeks - topic_agg[username]["radius"][::-1].index(None)
        idx = total_weeks - topic_agg[username]["anomaly_doc_ids"][::-1].index(None)
    except ValueError:
        idx = 0


    t_idx = idx + 1
    x_range = range(t_idx, total_weeks+1)
    axis_range = [0, total_weeks+1, -0.1, 1.1]
    print "week %d: %s" % (t_idx, username)


    try:
        f, a =  plt.subplots(total_weeks - idx, 1, sharex=True, sharey=True, figsize=(8.5, 20))
        #a = a.ravel()

        for t, ax in enumerate(a):
            indegree_data = topic_agg[username]['doc_id_indegree'][idx+t].values()
            ax.hist(indegree_data, bins=10)
            title = "week%d" % (t_idx+t)
            ax.set_title(title)

        f.savefig('%s/distr_w%d_%s.png' % (figure_output, t_idx, username))
    except TypeError:
        print "TypeError %s" % username
        continue

    fig = plt.figure(figsize=(8.5, 10.8))
    fig.suptitle('Anomaly User Detection: %s, Score:%d' % (username, priority[username]), fontsize=16, fontweight='bold')

    ax = fig.add_subplot(211)
    ax.set_axis_bgcolor('w')
    ax.grid(color='k', linestyle='dotted', linewidth=1)
    ax.set_title('Topic Dispersion')
    ax.set_ylabel('value')
    ax.axis(axis_range)

    # plot radius
    radius_values = map(lambda x: len(x), topic_agg[username]['anomaly_doc_ids'][idx:])

    if np.sum(radius_values) != 0:
        radius_ratio = np.array(radius_values, dtype=float) / np.sum(radius_values)
    else: 
        radius_ratio = np.array(radius_values, dtype=float)

    print radius_values, radius_ratio
    line_radius, = ax.plot(x_range, radius_ratio)
    for y_idx, x in enumerate(x_range):
        ax.text(x, radius_ratio[y_idx]-0.05, '%.3f' % radius_values[y_idx], color='blue')

    # plot uniq_doc
    #uniq_doc_values = topic_agg[username]['uniq_doc'][idx:]
    uniq_doc_values = map(lambda x: len(x),topic_agg[username]['doc_id_indegree'][idx:])
    uniq_doc_ratio = np.array(uniq_doc_values, dtype=float) / sum(uniq_doc_values)
    line_uniq_doc, = ax.plot(x_range, uniq_doc_ratio)
    # annotate real uniq doc counts
    for y_idx, x in enumerate(x_range):
        ax.text(x, uniq_doc_ratio[y_idx], uniq_doc_values[y_idx])

    ax.legend([line_radius, line_uniq_doc], ['topic radius', 'uniq docs access'], bbox_to_anchor=(1.1, 1.13), borderaxespad=0.)


    # the user be filtered in user cluster analysis
    try:
        idx = total_weeks - user_agg[username]["distance"][::-1].index(None)
    except ValueError:
        idx = 0

    u_idx = idx + 1
    x_range = range(u_idx, total_weeks+1)

    if user_agg[username]["distance"] == [] or x_range == []:
        fig.savefig('%s/%d_w%d_%s.png' % (figure_output, priority[username], t_idx, username))
        continue

    ax = fig.add_subplot(212)
    ax.set_axis_bgcolor('w')
    ax.grid(color='k', linestyle='dotted', linewidth=1)
    ax.set_title('User Collaoration')
    ax.set_xlabel('week')
    ax.set_ylabel('value')
    ax.axis(axis_range)

    # plot distance 
    distances = user_agg[username]["distance"][idx:]
    line_distance, = ax.plot(x_range, distances)
    for y_idx, x in enumerate(x_range):
        ax.text(x, distances[y_idx]-0.05, '%.3f' % distances[y_idx], color='blue')

    # plot uniq_doc
    cluster_size_list = [len(cluster) for cluster in user_agg[username]["filtered_users"] if cluster is not None]
    cluster_size_ratio = np.array(cluster_size_list, dtype=float) / sum(cluster_size_list)
    line_cluster_size, = ax.plot(x_range, cluster_size_ratio)
    # annotate real uniq doc counts
    for y_idx, x in enumerate(x_range):
        ax.text(x, cluster_size_ratio[y_idx], cluster_size_list[y_idx])

    # plot uniq_doc
    anonymous_rank = []
    anonymous_score = []
    for cid, cluster in enumerate(user_agg[username]["filtered_users"]):
        if cluster is None: continue

        if '-' in cluster:
            anonymous_rank.append(cluster.index('-')+1)
            anonymous_score.append(user_agg[username]["filtered_score"][cid]['-'])
        else:
            anonymous_rank.append(0)
            anonymous_score.append(0)
            
    
    if np.sum(anonymous_score) != 0:
        anonymous_score_ratio = np.array(anonymous_score, dtype=float) / np.sum(anonymous_score)
    else: 
        anonymous_score_ratio = np.array(anonymous_score, dtype=float) 

    print anonymous_score, anonymous_score_ratio
    line_anonymous_score, = ax.plot(x_range, anonymous_score_ratio)
    # annotate real uniq doc counts
    for y_idx, x in enumerate(x_range):
        ax.text(x, anonymous_score_ratio[y_idx], anonymous_rank[y_idx])

    ax.legend([line_distance, line_cluster_size, line_anonymous_score], ['variety score', 'cluster size', 'anonymous score'], bbox_to_anchor=(1.1, 1.13), borderaxespad=0.)

    fig.savefig('%s/%d_w%d_%s.png' % (figure_output, priority[username], t_idx, username))





