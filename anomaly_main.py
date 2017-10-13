from lib.anomaly_user import AnomalyUserDetection as AUD
from lib.selection import ElasticColumn
from datetime import datetime, timedelta

start = datetime(2016, 11, 5)
for w in range(1, 9)[::-1][:1]:
    c = start + timedelta(days=7*w)
    aud = AUD(c.year, c.month, c.day, unit=7, beta=3, wsize=0, force_start=4)
    aud.exp4_query('someone')


start = datetime(2016, 11, 10)
length = 60

for d in range(length):
    current = start+timedelta(days=d)
    print "title: %s" % current.strftime("%Y/%m/%d")
    aud = AUD(current.year, current.month, current.day, unit=1, beta=1.1, wsize=0, force_start=4)
    aud.exp4_query('another one')


