
# decorator pattern
class LegoProcess(object):
    def __init__(self, process_obj):
        self.__process_obj = process_obj        

    def process(self, value): raise NotImplementedError

    # next function to check object exist
    def next(self, value):
        if self.__process_obj:
            return self.__process_obj.process(value)

        return None
