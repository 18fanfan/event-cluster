from sklearn.externals import joblib
import os, numpy as np, sys

vsm = '/tmp/event_analaysis_output/modeling/TfIdfMatrix_False_False_doc_matrix_term_2016-11-07_2017-01-01.model'
index = joblib.load(vsm)
feature_m = index["matrix"]

for dirpath, dirnames, filenames in os.walk("/tmp/event_analaysis_output/evaluation/"):
    for filename in filenames:
        if "2016-11-07_2017-01-01" in filename:
            path = "%s%s" % (dirpath, filename)
            model = joblib.load(path)
            lda = model["fitted_model"]

            population_size = feature_m.T.shape[0]
            sample_rate, sample_count = 0.05, 50
            sample_size = population_size * sample_rate
            point_estimations = []

            for i in range(sample_count):
                samples = feature_m.T[np.random.choice(population_size, size=sample_size, replace=False),:]
                p_without_dist = lda.perplexity(samples)
                point_estimations.append(p_without_dist)
                print i, p_without_dist
                sys.stdout.flush()
                

            print "%s: p_without_dist=%.5e" % (filename, np.average(point_estimations))

