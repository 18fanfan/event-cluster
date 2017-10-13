import unittest, sys, shutil, os
from contextlib import contextmanager
from cStringIO import StringIO
from lib.usage_miner import UsageMiner
from datetime import datetime, timedelta
from lib.selection import *
from lib.phase_interface import PhaseInterface


@contextmanager
def capture(command, *args, **kwargs):
    tmp_out, sys.stdout = sys.stdout, StringIO()
    try:
        command(*args, **kwargs)
        sys.stdout.seek(0)
        yield sys.stdout.read() 
    finally:
        # contextlib tear down
        sys.stdout = tmp_out


class TestUsageMiner(unittest.TestCase):

    def setUp(self):
        self.start_date = datetime(2017, 1, 3)
        self.offset_days = 5
        self.output_base = "/tmp/test_without_strategy"

    def tearDown(self):
        if os.path.exists(self.output_base):
            shutil.rmtree(self.output_base)

    def test_usage_miner_wrong_new(self):
        # no start_date and offset
        self.assertRaises(TypeError, UsageMiner)

    def test_usage_miner_without_strategy(self):
        um = UsageMiner(self.start_date, self.offset_days, output_base = self.output_base)

        with capture(um.mining) as output:
            self.assertIn("phase 1: data_selection has no result. stop the next phase.", output)
            self.assertIn("pass", output)

        self.assertTrue(os.path.exists(self.output_base))
    

class TestDataSelection(unittest.TestCase):

    def setUp(self):
        self.base = "/tmp/%s" % self.__class__.__name__
        self.output_base = "/tmp/%s/data_selection" % self.__class__.__name__
        if not os.path.exists(self.output_base):
            os.makedirs(self.output_base)

    def tearDown(self):
        if os.path.exists(self.base):
            shutil.rmtree(self.base)

    def __touch_file(self, filepath):
        with open(filepath, 'w') as f:
            f.write("test")

    def test_PhaseInterface(self):
        self.assertRaises(NotImplementedError, PhaseInterface().run, None, None)
        self.assertRaises(NotImplementedError, PhaseInterface().check_result, None, None)

    def test_ElasticColumn_check_result_one_offset_days_return_None(self):
        start_date = datetime(2017, 1, 3)
        offset_days = 1
        query = ""
        columns = ["col1", "col2"] 
        actual = ElasticColumn(query, columns).check_result((start_date, offset_days), self.output_base)
        self.assertEqual(actual, None)

    def test_ElasticColumn_check_result_zero_offset_days_return_list(self):
        start_date = datetime(2017, 1, 3)
        offset_days = 0
        query = ""
        columns = ["col1", "col2"] 

        datestr = start_date.strftime("%Y-%m-%d")
        pattern = '_'.join(["ElasticColumn"] + columns + [datestr])
        filepath = "%s/%s" % (self.output_base, pattern)
        self.__touch_file(filepath)

        expect = [filepath]
        actual = ElasticColumn(query, columns).check_result((start_date, offset_days), self.output_base)
        self.assertEqual(actual, expect)

    def test_ElasticColumn_check_result_five_offset_days_return_None(self):
        start_date = datetime(2016, 12, 30)
        offset_days = 5
        query = ""
        columns = ["col1", "col2"] 
        actual = ElasticColumn(query, columns).check_result((start_date, offset_days), self.output_base)
        self.assertEqual(actual, None)

    def test_ElasticColumn_check_result_five_offset_days_with_parital_file_return_None(self):
        start_date = datetime(2016, 12, 30)
        offset_days = 5
        query = ""
        columns = ["col1", "col2"] 
        for d in range(offset_days - 2):
            datestr = (start_date + timedelta(days=d)).strftime("%Y-%m-%d")
            pattern = '_'.join(["ElasticColumn"] + columns + [datestr])
            filepath = "%s/%s" % (self.output_base, pattern)
            self.__touch_file(filepath)

        actual = ElasticColumn(query, columns).check_result((start_date, offset_days), self.output_base)
        self.assertEqual(actual, None)

    def test_ElasticColumn_check_result_five_offset_days_with_parital_file_return_list(self):
        start_date = datetime(2016, 12, 30)
        offset_days = 5
        query = ""
        columns = ["col1", "col2"] 
        expect = []
        for d in range(offset_days + 1):
            datestr = (start_date + timedelta(days=d)).strftime("%Y-%m-%d")
            pattern = '_'.join(["ElasticColumn"] + columns + [datestr])
            filepath = "%s/%s" % (self.output_base, pattern)
            self.__touch_file(filepath)
            expect.append(filepath)

        actual = ElasticColumn(query, columns).check_result((start_date, offset_days), self.output_base)
        self.assertEqual(actual, expect)

    def test_ElasticColumn_run_check_line_num_return_list(self):
        start_date = datetime(2016, 12, 31)
        offset_days = 1
        query = "type:apache_access AND company:tbox AND rawrequest: ((\"GET\" AND (\"view\" OR \"cp\" OR \"preview\" OR \"?a\"))  OR (\"PUT\" OR \"FILEPATCH\" OR \"DELTA\") AND (NOT \"FinishedPut\")) AND (NOT tags:\"_grokfailure\")"
        columns = [
            "@timestamp",
            "auth",
            "rawrequest",
            "agent",
            "clientip"
        ]
        expect = []
        for d in range(offset_days + 1):
            datestr = (start_date + timedelta(days=d)).strftime("%Y-%m-%d")
            pattern = '_'.join(["ElasticColumn"] + columns + [datestr])
            filepath = "%s/%s" % (self.output_base, pattern)
            expect.append(filepath)
        

        actual = ElasticColumn(query, columns).run(None, self.output_base, start_date, offset_days)
        self.assertEqual(actual, expect)

        for filepath in actual:
            with open(filepath, 'r') as f:
                line_count = len(f.readlines())
                self.assertGreater(line_count, 5200)


if __name__ == '__main__':
    unittest.main()
    #td = TestDataSelection()
    #td.setUp()
    #td.test_ElasticColumn_check_result_one_offset_days_return_list()
    #td.tearDown()
