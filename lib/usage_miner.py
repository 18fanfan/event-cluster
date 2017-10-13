# -*- encoding: utf-8 -*-
import datetime

# strategy design pattern
class UsageMiner(object):

    def __init__(self, init_ctx, \
        # TODO configuration
        force_start = None, \
        selection_obj = None,  \
        preprocessing_obj = None, \
        transformation_obj = None, \
        modeling_obj = None, \
        evaluation_obj = None):

        self.__init_ctx = init_ctx
        # setup strategy object for each phase
        self.data_selection_obj = selection_obj
        self.data_preprocessing_obj = preprocessing_obj
        self.data_transformation_obj = transformation_obj
        self.modeling_obj = modeling_obj
        self.evaluation_obj =  evaluation_obj

        self.__force_start = force_start
        # prepare output directory

    def mining(self):
        phases = [
            "data_selection",
            "data_preprocessing",
            "data_transformation",
            "modeling",
            "evaluation"
        ]

        # init input
        phase_ctx = self.__init_ctx
        for idx, phase in enumerate(phases):
            print "[%s] phase %d: %s start" % (self.__get_current_time(), idx+1, phase)

            is_result_exists = False
            if self.__force_start is None or self.__force_start > (idx + 1):
                # normal check
                next_ctx = self.__do_phase("check_result", phase, phase_ctx)
                if next_ctx is not None:
                    is_result_exists = True
                    print "[%s] phase %d: %s find result and reuse it." % (self.__get_current_time(), idx+1, phase)

            if not is_result_exists:
                # force run this phase or result does not exists
                next_ctx = self.__do_phase("run", phase, phase_ctx)
                if next_ctx is None: 
                    print "[%s] phase %d: %s has no result. stop the next phase." % (self.__get_current_time(), idx+1, phase)
                    break 

            phase_ctx = next_ctx
            print "[%s] phase %d: %s done" % (self.__get_current_time(), idx+1, phase)

        print "-" * 100
        return phase_ctx

    def __get_current_time(self):
        return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def __do_phase(self, action, phase, phase_ctx):
        phase_obj = getattr(self, "%s_obj" % phase, None)
        if phase_obj:
            start = datetime.datetime.now()
            ret = getattr(phase_obj, action)(phase, **phase_ctx)
            elapsed_time = datetime.datetime.now() - start
            print "elapsed time: %.2f ms" % ((elapsed_time.seconds * 1000) + (elapsed_time.microseconds / 1000.0))
            return ret

        return None
