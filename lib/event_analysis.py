# -*- encoding: utf-8 -*-
from usage_miner import UsageMiner
from datetime import datetime
from lib.selection import ElasticColumn
from lib.preprocessing import FieldFilterAndProcess
from lib.field_process import *
from lib.transformation import UserTermDocIndex
from lib.modeling import TfIdfMatrix
from lib.model_process import RetainLSA, LeftRetainLSA 
from lib.evaluation import *
import numpy as np

# Facade pattern
class EventAnalysis(object):

    def __init__(self, year, month, day, offset_days, output_base = "/tmp/event_analaysis_output", force_start = None):
        self.__start_date = datetime(year, month, day)
        self.__offset_days = offset_days
        self.__output_base = output_base
        self.__force_start = force_start
        self.__data_selection = None
        self.__data_preprocessing = None
        self.__data_transformation = None
        self.__modeling = None
        self.__evaluation = None
            
    def mining(self):
        #setup a new miner
        init_ctx = {
            "base": self.__output_base,
            "phase_input": (self.__start_date, self.__offset_days)
        }
        miner = UsageMiner(
            init_ctx, 
            force_start = self.__force_start,
            selection_obj = self.__data_selection, 
            preprocessing_obj = self.__data_preprocessing, 
            transformation_obj = self.__data_transformation,
            modeling_obj = self.__modeling,
            evaluation_obj = self.__evaluation
        )
        return miner.mining()

    def __setup(self):
        query = "type:apache_access AND company:tbox AND rawrequest:(((GET AND \/api\/v2\/) OR (GET AND \/api\/v1\/) OR (GET AND \/dav\/) OR PUT OR FILEPATCH OR DELTA) AND (NOT FinishedPut) AND (NOT SIGNATURE) AND (NOT MKCOL) AND (NOT PROPFIND) AND (NOT DELETE) AND (NOT FASTPUT) AND (NOT HEAD)) AND (NOT tags:'_grokfailure')"
        columns = [
            "@timestamp",
            "auth",
            "rawrequest",
            "agent",
            "clientip"
        ]
        p1 = ElasticColumn(query, columns)
        rawrequest_process = List2Scalar()
        rawrequest_process = RawRequestFilter(process_obj=rawrequest_process)
        rawrequest_process = ColumnFilter([1, 2], process_obj=rawrequest_process)
        rawrequest_process = CleanAccessPath(process_obj=rawrequest_process)
        rawrequest_process = Tokenizer(process_obj=rawrequest_process)
        rawrequest_process = RemoveStopWord(process_obj=rawrequest_process)
        column_process = {
            "rawrequest": rawrequest_process,
            "auth": List2Scalar()
        }  
        p2 = FieldFilterAndProcess(column_process)
                    
        mapping = {
            "user": "auth",
            "term": "rawrequest"
        }
        p3 = UserTermDocIndex(mapping)
        self.set_data_selection(p1)
        self.set_data_preprocessing(p2)
        self.set_data_transformation(p3)

    def exp1_cluster_8_tfidf_normalized(self):
        self.__setup()
        p4 = TfIdfMatrix()
        p5 = KMeansCluster(20)
        self.set_modeling(p4)
        self.set_evaluation(p5)
        self.mining()

    def exp2_cluster_8_tfidf_without_normalized(self):
        self.__setup()
        p4 = TfIdfMatrix(normalize=False)
        p5 = KMeansCluster(8, normalize=False)
        self.set_modeling(p4)
        self.set_evaluation(p5)
        self.mining()

    def exp3_cluster_8_tfidf_normalized_retainLSA_100_normalized(self):
        self.__setup()
        p4 = TfIdfMatrix(normalize=True)
        p5 = KMeansCluster(50, normalize=True, approx=RetainLSA(normalize=True))
        self.set_modeling(p4)
        self.set_evaluation(p5)
        self.mining()

    def exp4_cluster_8_tfidf_retainLSA_100(self):
        self.__setup()
        p4 = TfIdfMatrix(normalize=False)
        p5 = KMeansCluster(50, normalize=False, approx=RetainLSA(normalize=False))
        self.set_modeling(p4)
        self.set_evaluation(p5)
        self.mining()

    def exp5_cluster_8_tfidf_normalized_LeftRetainLSA_100_normalized(self):
        self.__setup()
        p4 = TfIdfMatrix()
        p5 = KMeansCluster(8, normalize=True, approx=LeftRetainLSA())
        self.set_modeling(p4)
        self.set_evaluation(p5)
        self.mining()

    def exp6_cluster_8_tfidf_normalized_retainLSA_100(self):
        self.__setup()
        p4 = TfIdfMatrix(normalize=True)
        p5 = KMeansCluster(50, normalize=True, approx=RetainLSA(normalize=False))
        self.set_modeling(p4)
        self.set_evaluation(p5)
        self.mining()

    def exp7_lda(self):
        self.__setup()
        p4 = TfIdfMatrix(normalize=False, idf=False)
        p5 = LDA(10, alpha=0.01, beta=0.01)
        self.set_modeling(p4)
        self.set_evaluation(p5)
        self.mining()

    def exp8_lda(self):
        self.__setup()
        p4 = TfIdfMatrix(normalize=False, idf=False)
        p5 = LDA(10, alpha=1, beta=1)
        self.set_modeling(p4)
        self.set_evaluation(p5)
        self.mining()

    def exp9_lda(self):
        self.__setup()
        p4 = TfIdfMatrix(normalize=False, idf=False)
        p5 = LDA(100, alpha=1, beta=1)
        self.set_modeling(p4)
        self.set_evaluation(p5)
        self.mining()

    def exp10_lda(self):
        self.__setup()
        p4 = TfIdfMatrix(normalize=False, idf=False)
        p5 = LDA(100, alpha=0.1, beta=0.1)
        self.set_modeling(p4)
        self.set_evaluation(p5)
        self.mining()

    def exp11_lda(self):
        self.__setup()
        p4 = TfIdfMatrix(normalize=False, idf=False)
        p5 = LDA(100, alpha=0.01, beta=0.01)
        self.set_modeling(p4)
        self.set_evaluation(p5)
        self.mining()

    def exp12_lda(self):
        self.__setup()
        p4 = TfIdfMatrix(normalize=False, idf=False)
        p5 = LDA(10, alpha=0.1, beta=0.1)
        self.set_modeling(p4)
        self.set_evaluation(p5)
        self.mining()

    def exp13_lda(self):
        self.__setup()
        p4 = TfIdfMatrix(normalize=False, idf=False)
        p5 = LDA(100, alpha=0.1, beta=0.01)
        self.set_modeling(p4)
        self.set_evaluation(p5)
        self.mining()

    def exp14_topic_dev_topic_feature(self):

        def std(distances):
            return np.sqrt(np.average(distances))

        def cv(distances):
            mean = np.average(distances)
            if mean > 0.0:
                return np.std(np.array(distances)) / mean

            return 0.0

        # another diameter
        def avg_link(distances):
            if len(distances) == 0:
                return 0.0

            return np.average(distances)

        def complete_link(distances):
            if len(distances) == 0:
                return 0.0

            return np.max(distances)

        def complete_link_cv(distances):
            if len(distances) == 0:
                return 0.0

            return np.max(distances) / np.average(distances)

            
        self.__setup()
        p4 = TfIdfMatrix(normalize=False, idf=False)
        p5 = TopicDispersion(100, word_feature=False, dispersion_func=cv, alpha=0.1, beta=0.01)
        self.set_modeling(p4)
        self.set_evaluation(p5)
        return self.mining()

    def exp15_knn_graph(self):
        self.__setup()
        p4 = TfIdfMatrix(normalize=False, idf=False)
        #p5 = KNNGraph(100, word_feature=False, alpha=0.1, beta=0.01)
        p5 = KNNGraph(100, word_feature=False, alpha=0.01, beta=0.01, t=2, uniq_file=9999999999)
        self.set_modeling(p4)
        self.set_evaluation(p5)
        return self.mining()


    def set_data_selection(self, value):
        self.__data_selection = value

    def set_data_preprocessing(self, value):
        self.__data_preprocessing = value

    def set_data_transformation(self, value):
        self.__data_transformation = value

    def set_modeling(self, value):
        self.__modeling = value

    def set_evaluation(self, value):
        self.__evaluation = value

