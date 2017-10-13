import ldif, codecs, sys, json, os
from collections import deque, defaultdict
import pprint 



class ClusterScore(object):

    def __init__(self, ldif_input):
        basename = os.path.basename(ldif_input)
        filename ,_ = os.path.splitext(basename)
        ds_output = "/tmp/tree_group_%s.json" % filename
        self.__data = None
        self.__ldif_input = ldif_input
        self.__tree_height = 0

        if not self.__is_build(ds_output):
            self.__parse_ldif(ldif_input)
            self.__build_tree_and_group("/tmp/tree_group_%s.json" % filename)

        self.__tree_check()
        self.__group_check()


    def topk_group(self, input_group, condition=None, k=10, weight=1):
        result = []
        for dn, group in self.__group.items():
            if self.__meet_condition(condition, group):
                result.append((dn, self.__weighted_f1_score(group, input_group, weight)))

        return sorted(result, key=lambda x: x[1], reverse=True)[:k]
            
    def node_distance(self, a, b):
        lca = self.__lca(a, b)
        if lca is None:
            return sys.maxint, lca
        
        return self.__distance(a)+self.__distance(b)-(2*self.__distance(lca)), lca


    def get_group(self):
        return self.__group


    def norm_node_distance(self, a, b):
        #NOTE: Eva usually the root of trenders not steve chang
        distance, lca = self.node_distance(a, b)
        return  distance/(2.0 * (self.__tree_height - 1)), lca

    def get_tree_height(self):
        return self.__tree_height


    def get_ad_data(self):
        if self.__data is None:
            print "lasy parse ad data"
            self.__parse_ldif(self.__ldif_input)

        return self.__data


    def __parse_ldif(self, ldif_input):
        parser = ldif.LDIFRecordList(codecs.open(ldif_input, 'r', encoding='utf8'))
        print "parse ldif file: %s" % ldif_input
        parser.parse()
        print "convert parse result to dictionary"
        self.__data = dict(map(lambda (k, v): (k.decode('utf8'), v), parser.all_records))
        

    def __is_build(self, ds_output):
        if os.path.exists(ds_output):
            print "find cache and load: %s" % ds_output
            ds = json.load(codecs.open(ds_output, 'r', encoding='utf8'))
            self.__tree = ds['tree']
            self.__group = ds['group']
            return True

        return False

        
    def __build_tree_and_group(self, ds_output):
            
        print "build tree and group"
        self.__group, self.__tree = defaultdict(list), dict()
        count = 0
        for dn, obj in self.__data.items():
            if 'Person' in obj['objectCategory'][0]:
                # process organization
                node_name = obj['sAMAccountName'][0]
                node_name = node_name.lower()

                if not 'manager' in obj:
                    print "[%s] has no manager attribute" % node_name
                    self.__tree[node_name] = None
                    continue
                
                if not obj['manager'][0] in self.__data:
                    print "[%s]'s manager is not in data" % node_name
                    self.__tree[node_name] = None
                    continue

                parent_node_name = self.__data[obj['manager'][0]]['sAMAccountName'][0]
                parent_node_name = parent_node_name.lower()
                if node_name in self.__tree:
                    print "login name duplication: %s" % node_name
                    continue

                if node_name == parent_node_name:
                    self.__tree[node_name] = None
                else:
                    self.__tree[node_name] = parent_node_name
                
            elif 'Group' in obj['objectCategory'][0]:
                # process group
                queue = deque([dn])
                seen = set()

                while len(queue) > 0:
                    cur_dn = queue.popleft()
                    if not cur_dn in self.__data: continue

                    if 'Person' in self.__data[cur_dn]['objectCategory'][0]:
                        self.__group[dn].append(self.__data[cur_dn]['sAMAccountName'][0].lower())
                    elif 'Group' in self.__data[cur_dn]['objectCategory'][0]:
                        if not 'member' in self.__data[cur_dn]: continue
                        if cur_dn in seen: continue

                        if cur_dn in self.__group: 
                            self.__group[dn] += self.__group[cur_dn]
                            continue
                    
                        for member_dn in self.__data[cur_dn]['member']:
                            queue.append(member_dn)

                        seen.add(cur_dn)
                
            count += 1
            print count, len(self.__data)

        ds = {
            'tree': self.__tree,
            'group': self.__group
        }
        json.dump(ds, codecs.open(ds_output, 'w', encoding='utf-8'))
            
    def __tree_check(self):
        print "tree check"
        number_of_nodes = len(self.__tree)
        number_of_edges = 0
        edge_dict = defaultdict(int)
        for nodename in self.__tree:
            currname = nodename
            while self.__tree[currname]:
                #NOTE it comes infinite loop if cycle exists
                if edge_dict[currname] == 1:
                    break

                edge_dict[currname] = 1
                number_of_edges += 1
                currname = self.__tree[currname]
        
        seen_root = set()
        for nodename in self.__tree:
            curr_path = self.__step_to_root(nodename)

            if curr_path == []: 
                print "In impossible situation due to nodename should exist in tree"
                exit(1)

            if self.__tree_height < len(curr_path) - 1:
                self.__tree_height = len(curr_path) - 1

            curr_root = curr_path[-1]
            seen_root.add(curr_root)
        
        if number_of_nodes != number_of_edges + len(seen_root):
            print "a graph in the forest"
            print "number of nodes: %d, number of edges: %d, edge+forest: %d" % (number_of_nodes, number_of_edges, number_of_edges + len(seen_root))
            exit(1)
        else:
            print "all trees in forest"
            print "number of nodes: %d, number of edges: %d, edge+forest: %d" % (number_of_nodes, number_of_edges, number_of_edges + len(seen_root))


    def __step_to_root(self, nodename):
        if nodename not in self.__tree:
            return []

        currname = nodename
        path = [currname]
        while self.__tree[currname]:
           currname = self.__tree[currname]
           path.append(currname)

        return path

    def __distance(self, n):
        return len(self.__step_to_root(n)) - 1

    def __group_check(self):
        dup = 0
        for dn, group in self.__group.items():
            if len(group) != len(set(group)):
                dup += 1

        print "number of groups(DL): %d, number of member duplication group: %d" % (len(self.__group), dup)

    def __lca(self, a, b):
        path1 = self.__step_to_root(a)[::-1]
        path2 = self.__step_to_root(b)[::-1]

        lca = None
        for i in range(min(len(path1), len(path2))):
            if path1[i] != path2[i]:
                break
            else:
                lca = path1[i]
        
        return lca

    def __weighted_f1_score(self, truth, result, weight):
        if (len(truth) & len(result)) == 0: return 0
        truth_set = set(truth)
        result_set = set(result)
        inter = truth_set.intersection(result_set)
        p = len(inter) / float(len(result_set))
        r = len(inter) / float(len(truth_set))
        if p+r == 0.0: return 0
        return ((weight+1)*r*p)/((weight)*r+p)

    def __meet_condition(self, condition, validate):
        if condition is None: return True
        if set(validate).issuperset(set(condition)):
            return True

        return False




if __name__ == '__main__':
    # remove all the test case due to the confidential
