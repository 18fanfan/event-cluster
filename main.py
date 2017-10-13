from lib.event_analysis import EventAnalysis
from lib.selection import ElasticColumn
from lib.field_process import List2Scalar, RawRequestFilter, ColumnFilter, MethodAccessTransform, CleanAccessPath, Tokenizer
from lib.preprocessing import FieldFilterAndProcess
import  cPickle, hashlib, datetime

for i in range(1, 9)[:1]:
    ea = EventAnalysis(2016, 11, 6, i*7-1, force_start=None)
    ea = EventAnalysis(2016, 11, 8, 0, force_start=5)
    ret = ea.exp14_topic_dev_topic_feature()
    print ret


y, m = 2016, 12
offset = 1
for d in range(29, 30):
    ea = EventAnalysis(y, m, d, offset, force_start=5)
    for method in [method for method in dir(ea) if callable(getattr(ea, method)) and method.startswith("exp")]:
        if reduce(lambda a, b: a | b, map(lambda exp_name: method.startswith(exp_name), white_list)):
            print "title: %s, date range %d-%d-%d~%d" % (method, y, m, d, offset)
            getattr(ea, method)()
       
