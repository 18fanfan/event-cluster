from phase_interface import FileBasedProcess, MANY2ONE
import os, hashlib, json, math, pandas, numpy as np, codecs
from collections import defaultdict

class BuildBipartiteGraph(FileBasedProcess):

    def __init__(self, type_mapping, unit, beta, wsize, undirected=False):
        super(BuildBipartiteGraph, self).__init__(MANY2ONE)
        self.__type_mapping = type_mapping
        self.__unit = unit
        self.__beta = beta
        self.__wsize = wsize
        self.__undirected = undirected

    def output_prop(self):
        return {
            "description": sorted(self.__type_mapping.values()) + [self.__unit, self.__beta, self.__wsize, self.__undirected], "ext": "json"
        }
        
    @FileBasedProcess.preparation
    def run(self, phase, **phase_ctx): 
        phase_input = phase_ctx.get("phase_input")
        result_path = phase_ctx.get("result_paths")[0]

        # consider wsize is whole phase_input
        if self.__wsize == 0:
            self.__wsize = len(phase_input)

        user, request = self.__type_mapping["type1"], self.__type_mapping["type2"]
        user2access, access2user = defaultdict(float), defaultdict(float)
        alluser, allaccess = set(), set()

        for exp, s_idx in enumerate(range(len(phase_input), 0, -1)[::self.__unit][:self.__wsize][::-1]):
            weight = math.pow(self.__beta, exp)
            s, e = max(0, s_idx-self.__unit), s_idx
            print s, e, weight
            print phase_input[s:e]
            for filepath in phase_input[s:e]:
                with open(filepath, 'r') as f:
                    for line in f:
                        record = json.loads(line)
                        username = record[user]
                        if record[request] is None: continue
                        action, access_path = record[request]

                        if self.__undirected:
                            user2access[(username, access_path)] += weight
                        else:
                            # action, access_path 
                            if action == "down":
                                # access_path outcoming edge
                                access2user[(access_path, username)] += weight
                            elif action == "up":
                                # user outcoming edge
                                user2access[(username, access_path)] += weight

                        alluser.add(username)
                        allaccess.add(access_path)

        # TODO build adj matrix with pandas
        #upper_right = pandas.DataFrame(user2access).fillna(0.0)
        #upper_right.index, upper_right.columns
        #down_left = pandas.DataFrame(access2user).fillna(0,0)

        alluser, allaccess = sorted(list(alluser)), sorted(list(allaccess))
        if self.__undirected:
            upper_right = self.__build_adj_matrix(user2access, alluser, allaccess)
            down_left = upper_right.T
        else:
            upper_right = self.__build_adj_matrix(user2access, alluser, allaccess)
            down_left = self.__build_adj_matrix(access2user, allaccess, alluser)

        transform = {
            "upper_right": upper_right.tolist(),
            "down_left": down_left.tolist(),
            "rows": alluser,
            "cols": allaccess
        }

        with open(result_path, 'w') as dest:
            dest.write(json.dumps(transform))
            dest.flush()


    def __build_adj_matrix(self, data_dict, rows, cols):
        adj_m = np.zeros((len(rows), len(cols)), dtype=float)
        row_idx = dict([(name, i) for (i, name) in enumerate(rows)])
        col_idx = dict([(name, i) for (i, name) in enumerate(cols)])
        
        for (row_name, col_name), value in data_dict.items():
            adj_m[row_idx[row_name], col_idx[col_name]] = value

        return adj_m
        

class UserTermDocIndex(FileBasedProcess):

    def __init__(self, user_term_dict):
        super(UserTermDocIndex, self).__init__(MANY2ONE)
        self.__user_term_dict = user_term_dict
        self.__termdmt = ','

    def output_prop(self):
        return {
            "description": sorted(self.__user_term_dict.values()), "ext": "json"
        }
        
    @FileBasedProcess.preparation
    def run(self, phase, **phase_ctx): 
        phase_input = phase_ctx.get("phase_input")
        result_path = phase_ctx.get("result_paths")[0]
        transform = defaultdict(dict) 
        hash_func = hashlib.md5

        for file_idx, filepath in enumerate(phase_input):
            with open(filepath, 'r') as src:
                for line_idx, line in enumerate(src):
                    record = json.loads(line)
                    term_list = record[self.__user_term_dict.get("term")]
                    user_name = record[self.__user_term_dict.get("user")]

                    # compute doc id from content, consider the order of the terms
                    if term_list is None: continue
                        
                    term_list = [t.encode('utf-8') for t in term_list]
                    doc_id = hash_func(self.__termdmt.join(term_list)).hexdigest()

                    if user_name in transform["user"]:
                        transform["user"][user_name][doc_id] += 1
                    else:
                        transform["user"][user_name] = defaultdict(int)
                        transform["user"][user_name][doc_id] += 1


                    # using set to dedup term
                    for target_term in frozenset(term_list):
                        pos_list = [pos for pos, term in enumerate(term_list) if term == target_term]

                        if target_term in transform["term"]:
                            transform["term"][target_term][doc_id] = pos_list
                        else:
                            transform["term"][target_term] = {doc_id: pos_list}
                    
                    transform["doc"][doc_id] = defaultdict(dict)
                    transform["doc"][doc_id]["term_list"] = term_list
                    transform["doc"][doc_id]["rawdoc_idx"] = tuple([file_idx, line_idx])


        with codecs.open(result_path, 'w', 'utf-8') as dest:
            dest.write(json.dumps(transform))
            dest.flush()


