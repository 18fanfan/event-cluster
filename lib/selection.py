import os, json
from elasticsearch import helpers, Elasticsearch
from elasticsearch.exceptions import NotFoundError
from phase_interface import FileBasedProcess, INIT2MANY
from datetime import datetime, timedelta

class ElasticColumn(FileBasedProcess):

    def __init__(self, query, output_columns):
        super(ElasticColumn, self).__init__(INIT2MANY)
        self.__output_columns = output_columns
        self.__query = query
        self.__query_template = "queries/column_template"

    def output_prop(self):
        return {
            "description": sorted(self.__output_columns) + [self.md5pickle(self.__query)]
        }
        
    @FileBasedProcess.preparation
    def run(self, phase, **phase_ctx): 
        curr_input = phase_ctx.get("phase_input")
        result_paths = phase_ctx.get("result_paths")
        start_date, offset_days = curr_input

        es_query = json.load(open(self.__query_template, 'r'))
        # specific structure for this query template
        # TODO genralization?
        es_query["fields"] = self.__output_columns
        es_query["query"]["query"]["query_string"]["query"] = self.__query

        # TODO explicit default value
        es = Elasticsearch("http://http://10.1.193.189:9200/")
        for d in range(offset_days + 1):
            print "selecting: ", result_paths[d]
            query_index = self.__get_index_name(start_date, d)

            try:
                record_gen = helpers.scan(es, query=es_query, index=query_index)

                with open(result_paths[d], 'w') as t:
                    for record in record_gen:
                        t.write("%s\n" % json.dumps(record))

                    t.flush()
            except NotFoundError as e:
                print e
                with open(result_paths[d], 'w') as t:
                    tmp = {"sort": [191280], "_type": "apache_access", "_index": "ssfe-log-analysis-2017.04.09", "_score": None, "fields": {"@timestamp": ["2017-04-09T18:20:18.000Z", 1491762018000], "rawrequest": ["\"GET /api/v2/userinfo?_auth=&no_set_auth_cookie=1&_=1491611515641 HTTP/1.0\""], "clientip": ["150.70.94.98"], "auth": ["hank_hsieh"], "agent": ["\"Mozilla/5.0 (Windows NT 6.3; WOW64; Trident/7.0; rv:11.0) like Gecko\""]}, "_id": "AVtWzWYg0NGiPNsqh5ey"}
                    t.write(json.dumps(tmp)+"\n")
                    t.flush()


    def __get_index_name(self, start_date, day):
        current = start_date + timedelta(days=day)
        return "%s-%s" % ("ssfe-log-analysis", current.strftime("%Y.%m.%d"))


class ElasticColumnWithIdx(FileBasedProcess):

    def __init__(self, query, output_columns):
        super(ElasticColumn, self).__init__(INIT2MANY)
        self.__output_columns = output_columns
        self.__query = query
        self.__query_template = "queries/column_template"

    def output_prop(self):
        return {
            "description": sorted(self.__output_columns) + [self.md5pickle(self.__query)]
        }
        
    @FileBasedProcess.preparation
    def run(self, phase, **phase_ctx): 
        curr_input = phase_ctx.get("phase_input")
        result_paths = phase_ctx.get("result_paths")
        start_date, offset_days = curr_input

        es_query = json.load(open(self.__query_template, 'r'))
        # specific structure for this query template
        # TODO genralization?
        es_query["fields"] = self.__output_columns
        es_query["query"]["query"]["query_string"]["query"] = self.__query

        # TODO explicit default value
        es = Elasticsearch()
        for d in range(offset_days + 1):
            print "selecting: ", result_paths[d]
            query_index = self.__get_index_name(start_date, d)

            try:
                record_gen = helpers.scan(es, query=es_query, index=query_index)

                with open(result_paths[d], 'w') as t:
                    for record in record_gen:
                        t.write("%s\n" % json.dumps(record))

                    t.flush()
            except NotFoundError as e:
                print e
                with open(result_paths[d], 'w') as t:
                    tmp = {"sort": [191280], "_type": "apache_access", "_index": "ssfe-log-analysis-2017.04.09", "_score": None, "fields": {"@timestamp": ["2017-04-09T18:20:18.000Z", 1491762018000], "rawrequest": ["\"GET /api/v2/userinfo?_auth=&no_set_auth_cookie=1&_=1491611515641 HTTP/1.0\""], "clientip": ["150.70.94.98"], "auth": ["hank_hsieh"], "agent": ["\"Mozilla/5.0 (Windows NT 6.3; WOW64; Trident/7.0; rv:11.0) like Gecko\""]}, "_id": "AVtWzWYg0NGiPNsqh5ey"}
                    t.write(json.dumps(tmp)+"\n")
                    t.flush()


    def __get_index_name(self, start_date, day):
        current = start_date + timedelta(days=day)
        return "%s-%s" % ("ssfe-log-analysis", current.strftime("%Y.%m.%d"))
