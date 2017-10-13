import json, os
from sklearn.externals import joblib
import matplotlib.pyplot as plt
index = '/home/safesync/test/user_topic/data_transformation/7056dcd3bdf8a09c0fb8c59d129b7142_UserTermDocIndex_auth_rawrequest_2016-11-06_2017-01-07.json'

#index = '/home/safesync/tbox_3_to_5/user_topic/data_transformation/2a1f64b7a31174993df58d4cc4e86b33_UserTermDocIndex_auth_rawrequest_2017-03-12_2017-05-27.json'
print "load user term doc index"
index = json.load(open(index, 'r'))


user_topic_base = '/home/safesync/test/user_topic'
#user_topic_base = '/home/safesync/tbox_3_to_5/user_topic'
read_base = '%s/evaluation' % user_topic_base
print "load week evaluation"
weeks = [joblib.load("%s/%s" % (read_base, filename)) for filename in sorted(os.listdir(read_base), key=lambda x: x.split('_')[-1])]
print "number of weeks:", len(weeks)

topic_agg = dict()
user_number = 0
for idx, w in enumerate(weeks):
    for username, prop in w.items():
        if username in topic_agg:
            continue

        user_number += 1
        try:
            print "u%d, login_name: %s" % (user_number, username)
            print
            values = map(lambda x: (x[username]['anomaly_doc_ids'], x[username]['doc_id_indegree']), weeks[idx:])
            anomaly_ids_list, indegree_list = zip(*values)
            for num, doc_ids in enumerate(anomaly_ids_list):
                
                doc_ids_degree = sorted(map(lambda i: (i, indegree_list[num][i]), doc_ids), key=lambda x: x[1])
                for (doc_id, degree) in doc_ids_degree:
                    print "w%d, [%s], %d" % (num+idx+1, " ".join(index["doc"][doc_id]["term_list"]).encode('utf-8'), degree)

                print

            for num, (doc_id, degree) in enumerate(sorted(indegree_list[-1].items(), key=lambda x: x[1])):
                print "f%d, [%s], %d" % (num+1, " ".join(index["doc"][doc_id]["term_list"]).encode('utf-8'), degree)

            print

            topic_agg[username] = True
            print '-' * 60
        except KeyError:
            print "u%d, login_name: %s KeyError" % (user_number, username)
            print
            continue





