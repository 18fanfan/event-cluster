import os, cPickle, hashlib
from abc import ABCMeta, abstractmethod
from datetime import timedelta, datetime
from functools import partial
from pprint import PrettyPrinter

class PhaseInterface(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def run(self, phase, **phase_ctx): pass

    @abstractmethod
    def check_result(self, phase, **phase_ctx): pass


ONE2ONE = 0
MANY2ONE = 1
MANY2MANY = 2
INIT2MANY = 3

class FileBasedProcess(PhaseInterface):

    def __init__(self, check_type):
        self.__datedmt, self.__itemdmt = '-', '_'
        self.__check_type = check_type
        self.__check_result_dict = {
            ONE2ONE: self.__one2one,
            MANY2ONE: self.__many2one,
            MANY2MANY: self.__many2many,
            INIT2MANY: self.__init2many
        }
        self.__result_paths = None
    
    def run(self, phase, **phase_ctx): 
        raise NotImplementedError

    def output_prop(self):
        raise NotImplementedError

    @staticmethod
    def preparation(func):
        def output_wrapper(self, phase, **phase_ctx):
            result_base = self.__make_base(phase_ctx.get("base"), phase)
            if self.__result_paths:
                phase_ctx["result_paths"] = self.__result_paths
            else:
                result_base = "%s/%s" % (phase_ctx.get("base"), phase)
                phase_ctx["result_paths"] = self.__check_result_dict.get(self.__check_type)(phase_ctx.get("phase_input"), result_base)
            func(self, phase, **phase_ctx)
            phase_ctx["%s_input" % phase] = phase_ctx["phase_input"]
            phase_ctx["phase_input"] = phase_ctx["result_paths"]
            del phase_ctx["result_paths"]
            pp = PrettyPrinter(depth=4)
            print pp.pprint(phase_ctx)
            return phase_ctx

        return output_wrapper

    def check_result(self, phase, **phase_ctx):
        if phase_ctx.get("base") is None:
            raise ValueError("base directory not found")

        result_base = "%s/%s" % (phase_ctx.get("base"), phase)
        phase_input = phase_ctx.get("phase_input")
        self.__result_paths = self.__check_result_dict.get(self.__check_type)(phase_input, result_base)
        if False in map(os.path.exists, self.__result_paths):
            return None

        phase_ctx["%s_input" % phase] = phase_input
        phase_ctx["phase_input"] = self.__result_paths
        pp = PrettyPrinter(depth=4)
        print pp.pprint(phase_ctx)
        return phase_ctx

    def get_next(self):
        # TODO generalize file input/output operation 
        pass

    def output_result(self, result_obj):
        # TODO generalize file input/output operation 
        pass

    def md5pickle(self, input_obj):
        return hashlib.md5(cPickle.dumps(input_obj)).hexdigest()
        
    def __init2many(self, phase_input, result_base):
        start_date, offset_days = phase_input
        if type(start_date) is not datetime or type(offset_days) is not int:
            return None

        p_func = partial(self.__to_datestr, start_date)
        #including two end days, offset_days + 1
        datestrs = map(p_func, range(offset_days+1))
        p_func = partial(self.__get_result_path, result_base, self.output_prop())
        return map(p_func, datestrs, [[]]*len(datestrs))

    def __many2one(self, phase_input, result_base):
        if type(phase_input) is not list:
            return None

        date_ranges = sorted([item[2] for item in map(self.parse_filepath, phase_input)])
        first_date, last_date = date_ranges[0], date_ranges[-1]
        input_digest = self.md5pickle(sorted(phase_input))
        return [self.__get_result_path(result_base, self.output_prop(), first_date + last_date, input_digest)]

    def __many2many(self, phase_input, result_base):
        if type(phase_input) is not list:
            return None

        date_ranges = sorted([item[2] for item in map(self.parse_filepath, phase_input)])
        p_func = partial(self.__get_result_path, result_base, self.output_prop())
        
        input_digest_list = map(self.md5pickle, phase_input)
        return map(p_func, date_ranges, input_digest_list)

    def __one2one(self, phase_input, result_base):
        if type(phase_input) is not list and len(phase_input) != 1:
           return None

        date_ranges = sorted([item[2] for item in map(self.parse_filepath, phase_input)])
        p_func = partial(self.__get_result_path, result_base, self.output_prop())
        input_digest_list = map(self.md5pickle, phase_input)
        return map(p_func, date_ranges, input_digest_list)
        
    def __make_base(self, base, phase):
        path = "%s/%s" % (base, phase)
        self.__make_folder(path)
        return path

    def __make_folder(self, path):
        if not os.path.exists(path):
            os.makedirs(path)

        return path

    def __to_list(self, value):
        if type(value) is not list:
            return [str(value)]

        return map(str, value)

    def __to_datestr(self, start_date, day):
        return (start_date + timedelta(days=day)).strftime("%%Y%s%%m%s%%d" % (self.__datedmt, self.__datedmt))

    def __get_result_path(self, result_base, output_prop, date_range, input_digest):
        desc = output_prop.get("description")
        ext = "" if not output_prop.get("ext", False) else ".%s" % output_prop.get("ext")
        # add input_digest
        pattern = self.__itemdmt.join(self.__to_list(input_digest) + self.__to_list(self.__class__.__name__) + self.__to_list(desc) + self.__to_list(date_range))
        path = "%s/%s%s" % (result_base, pattern, ext)
        return path
        
    def parse_filepath(self, filepath):
        basename = os.path.basename(filepath)
        filename, _ = os.path.splitext(basename)
        items = filename.split(self.__itemdmt)
        class_name = items[0]
        items = items[1:]
        
        try:
            datetime.strptime(items[-2], "%%Y%s%%m%s%%d" % (self.__datedmt, self.__datedmt))
        except ValueError:
            date_range = items[-1:]
            items = items[:-1]
        else:
            date_range = items[-2:]
            items = items[:-2]
            
        desc = items
        return class_name, desc, date_range

