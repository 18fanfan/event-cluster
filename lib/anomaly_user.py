# -*- encoding: utf-8 -*-
from usage_miner import UsageMiner
from datetime import datetime
from lib.selection import ElasticColumn
from lib.preprocessing import FieldFilterAndProcess
from lib.field_process import *
from lib.transformation import BuildBipartiteGraph
from lib.modeling import RWRSM, RWRQuery
from lib.model_process import RetainLSA, LeftRetainLSA 
from lib.evaluation import *

class AnomalyUserDetection(object):

    def __init__(self, year, month, day, unit=1, beta=1.0, wsize=0, data_start=datetime(2016, 11, 6), output_base = "/tmp/anomaly_user", force_start = None):
        self.__target_date = datetime(year, month, day)
        self.__data_start = data_start
        self.__unit = unit
        self.__beta = beta
        self.__wsize = wsize

        self.__output_base = output_base
        self.__force_start = force_start
        self.__data_selection = None
        self.__data_preprocessing = None
        self.__data_transformation = None
        self.__modeling = None
        self.__evaluation = None

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

        # TODO adjust query by rawrequest filter
        rawrequest_process = List2Scalar()
        rawrequest_process = RawRequestFilter(process_obj = rawrequest_process)
        rawrequest_process = ColumnFilter([0, 2], process_obj = rawrequest_process)
        rawrequest_process = MethodAccessTransform(process_obj = rawrequest_process)
        column_process = {
            "rawrequest": rawrequest_process,
            "auth": List2Scalar()
        }  
        p2 = FieldFilterAndProcess(column_process)
        self.set_data_selection(p1)
        self.set_data_preprocessing(p2)
            
    def mining(self):
        #setup a new miner
        offset_days = (self.__target_date - self.__data_start).days
        init_ctx = {
            "base": self.__output_base,
            "phase_input": (self.__data_start, offset_days)

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

    def exp1_undirected(self):
        self.__setup()
        mapping = {
            "type1": "auth",
            "type2": "rawrequest"
        }

        p3 = BuildBipartiteGraph(mapping, self.__unit, self.__beta, self.__wsize, undirected=True)

        p4 = RWRSM()
        p5 = HeatMap()
        self.set_data_transformation(p3)
        self.set_modeling(p4)
        self.set_evaluation(p5)
        self.mining()

    def exp2_directed_topk(self):
        self.__setup()
        mapping = {
            "type1": "auth",
            "type2": "rawrequest"
        }

        p3 = BuildBipartiteGraph(mapping, self.__unit, self.__beta, self.__wsize, undirected=False)
        p4 = RWRSM()
        p5 = TopKUser(1000)
        self.set_data_transformation(p3)
        self.set_modeling(p4)
        self.set_evaluation(p5)
        return self.mining()

    def exp3_undirected_topk(self):
        self.__setup()
        mapping = {
            "type1": "auth",
            "type2": "rawrequest"
        }

        p3 = BuildBipartiteGraph(mapping, self.__unit, self.__beta, self.__wsize, undirected=True)
        p4 = RWRSM()
        p5 = TopKUser(1000)
        self.set_data_transformation(p3)
        self.set_modeling(p4)
        self.set_evaluation(p5)
        return self.mining()


    def exp4_query(self, query):
        self.__setup()
        mapping = {
            "type1": "auth",
            "type2": "rawrequest"
        }

        p3 = BuildBipartiteGraph(mapping, self.__unit, self.__beta, self.__wsize, undirected=False)
        p4 = RWRQuery(query)
        p5 = TopKUser(1000)
        self.set_data_transformation(p3)
        self.set_modeling(p4)
        self.set_evaluation(p5)
        self.mining()

    def rerun(self):
        self.__setup()
        self.mining()


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

