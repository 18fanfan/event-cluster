from sklearn.externals import joblib
import os

def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        print(" ".join([str((feature_names[i], topic[i]))
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))
    print

for dirpath, dirnames, filenames in os.walk('/tmp/event_analaysis_output/evaluation/'):
    for f in filenames:
        print f
        model = joblib.load("%s/%s" % (dirpath, f))
        lda = model["fitted_model"]
        terms = model["term"]
        print_top_words(lda, terms, 20)
            
    
