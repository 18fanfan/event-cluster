import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt, os, numpy as np
from sklearn.externals import joblib
from collections import defaultdict

topic_dev_base = '/home/safesync/presto-seg/event_cluster/topic_dev/evaluation/'
weeks = [joblib.load("%s/%s" % (topic_dev_base, filename)) for filename in sorted(os.listdir(topic_dev_base))]


total_weeks = len(weeks)
seen = dict()
for idx, w in enumerate(weeks):
    for username, prop in w.items():
        if seen.get(username, False):
            continue
        
        w_idx = idx+1
        print "week %d: %s" % (w_idx, username)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        fig.suptitle('Topic Deviation', fontsize=16, fontweight='bold')
        ax.set_title(str(username))
        ax.set_xlabel('week')
        ax.set_ylabel('value')
        ax.axis([0, total_weeks+1, -0.1, 1.1])
        values = map(lambda x: (x[username]['radius'], x[username]['uniq_doc']), weeks[idx:])
        radius_values, uniq_doc_values = zip(*values)

        x_range = range(w_idx, total_weeks+1)
        # plot radius
        line_radius, = ax.plot(x_range, radius_values)
        for y_idx, x in enumerate(x_range):
            ax.text(x, radius_values[y_idx]-0.03, '%.3f' % radius_values[y_idx], color='blue')

        # plot uniq_doc
        uniq_doc_ratio = np.array(uniq_doc_values, dtype=float) / sum(uniq_doc_values)
        line_uniq_doc, = ax.plot(x_range, uniq_doc_ratio)
        # annotate real uniq doc counts
        for y_idx, x in enumerate(x_range):
            ax.text(x, uniq_doc_ratio[y_idx], uniq_doc_values[y_idx])

        ax.legend([line_radius, line_uniq_doc], ['topic radius', 'uniq docs access'], bbox_to_anchor=(1., 1.13), borderaxespad=0.)


        fig.savefig('/tmp/topic_dev_result/w%d_%s.png' % (w_idx, username))
        seen[username] = True
        


