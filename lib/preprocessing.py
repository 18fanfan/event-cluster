import os, json
from phase_interface import FileBasedProcess, MANY2MANY

class FieldFilterAndProcess(FileBasedProcess):

    def __init__(self, column_process_methods):
        super(FieldFilterAndProcess, self).__init__(MANY2MANY)
        self.__process_methods = column_process_methods
        self.__columns = column_process_methods.keys()

    def output_prop(self):
        return {
            "description": sorted(self.__columns) + [self.md5pickle(self.__process_methods)]
        }
        
    @FileBasedProcess.preparation
    def run(self, phase, **phase_ctx): 
        curr_input = phase_ctx.get("phase_input")
        result_paths = phase_ctx.get("result_paths")

        for idx, filepath in enumerate(curr_input):
            with open(filepath, 'r') as src, open(result_paths[idx], 'w') as dest:
                print "preprocessing: %s" % result_paths[idx]
                for line in src:
                    # TODO exception handle
                    fields = json.loads(line)["fields"]
                    process_result = dict(map(lambda (k, method): (k, method.process(fields.get(k))), self.__process_methods.items()))
                    dest.write("%s\n" % json.dumps(process_result))
                    
                dest.flush()


