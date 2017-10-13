from lib.ad_knowledge.cluster_score import ClusterScore
from datetime import datetime
import os, json, codecs


ldif_input = "/tmp/somewhere"
topk_base = "/tmp/evaluation/"
cs = ClusterScore(ldif_input)

cluster_agg = dict()
cluster_scores = list()
user_tol = 4e-5
for dirpath, dirnames, filenames in os.walk(topk_base):
    for filename in filenames:
        basename, ext = os.path.splitext(filename)
        if ext != ".json": continue
        json_path = "%s%s" % (dirpath, filename)
        print json_path
        
        #31b3bd80787e1faee859103a8266aad9_TopKUser_1000_2017-03-11_2017-04-01.json
        basename_fields = basename.split('_')
        start_date, end_date = basename_fields[-2], basename_fields[-1]
        week = (datetime(*map(int, end_date.split('-'))) - datetime(*map(int, start_date.split('-')))).days / 7

        cluster_agg[week] = json.load(codecs.open(json_path, 'r', encoding='utf8'))
        
        for mastermind, cluster in cluster_agg[week].items():
            if mastermind == '-': continue
            valid_cluster = []
            for accomplice in cluster:
                if accomplice['prob'] <= user_tol: break
                
                if accomplice['name'] == mastermind:
                    valid_cluster.append((accomplice['name'], True))
                elif len(accomplice['accesses']) > 0:
                    valid_cluster.append((accomplice['name'], True))
                else:
                    valid_cluster.append((accomplice['name'], False))

            # only include himself
            if len(valid_cluster) == 1: continue

            diameter = []
            for accomplice_name, _ in valid_cluster:
#                if accomplice_name == '-': continue
                steps, lca = cs.node_distance(mastermind.lower(), accomplice_name.lower())
                diameter.append((accomplice_name, steps, lca))

            if len(diameter) <= 1:
                if len(diameter) == 0:
                    print "diameter 0", cluster, mastermind
                    exit(1)
                else:
                    continue

            condition = [valid_cluster[0][0]]
            input_group = map(lambda x: x[0], valid_cluster)
            best_group = cs.topk_group(input_group, condition=condition, k=1, weight=3)

            best_group_dn, score = None, -1
            if len(best_group) > 0:
                best_group_dn, score = best_group[0]

            cluster_scores.append({
                'diameter': diameter,
                'score': score,
                'best_group_dn': best_group_dn,
                'mastermind': mastermind,
                'week': week
            })


# print the aggregate result
ad_data = cs.get_ad_data()
sama2dn = dict(map(lambda (k, v): (v['sAMAccountName'][0].lower(), k), ad_data.items()))
p1_cols=30
p2_cols=30
p3_cols=60
p4_cols=40

def get_title(name):
    if name is None: return "None"
    dn = sama2dn.get(name.lower(), None)
    if dn is None:
        return "no samaname"
    else:
        data = ad_data.get(dn, None)
        if data is not None:
            title = data.get('title', None)
            if title is not None:
                return title[0]
            else:
                return "no title"
        else:
            return "no dn"

#    return ad_data[sama2dn[name]]['title']

for c_idx, cluster in enumerate(sorted(cluster_scores, key=lambda x: x['score'])):
    w = cluster['week']
    username = cluster['mastermind']
    avg_diameter = sum(map(lambda x: x[1], cluster['diameter'])) / float(len(cluster['diameter'])-1)
    valid_accomplice = map(lambda x: x[0], cluster['diameter'])

    print "#cluster%d: [mastermind:%s(%s), week:%d, score:%.3f]" % (c_idx+1, username, get_title(username), w, cluster['score'])
    print "\tvalid cluster:".ljust(p1_cols), "%s" % valid_accomplice
    print "\tuser cluster average diameter:".ljust(p1_cols) + "%.2f" % avg_diameter
    print "\teach accomplice's diameter:".ljust(p1_cols)
    for user_idx, (accomplice_name, steps, lca) in enumerate(cluster['diameter']):
        print ("\t\t%d: pair:[%s, %s(%s)]" % (user_idx+1, username, accomplice_name, get_title(accomplice_name))).ljust(p3_cols), ("lca:[%s(%s)]" % (lca, get_title(lca))).ljust(p4_cols), "steps:[%d]" % steps

    if cluster['best_group_dn'] is None:
        print "\tthe most similar group: None".ljust(p1_cols)
    else:
        group_member = (cs.get_group())[cluster['best_group_dn']]
        group_name = ad_data[cluster['best_group_dn']]['sAMAccountName']
        number_of_members = len(group_member)
        print ("\tthe most similar group(%d):" % number_of_members).ljust(p1_cols), "%s" % group_name
        inside = filter(lambda name: name in group_member, valid_accomplice)
        outside = filter(lambda name: name not in group_member ,valid_accomplice)
        print ("\tinside group members(%d):" % len(inside)).ljust(p1_cols), "%s" % inside
        print ("\toutside group members(%d):" % len(outside)).ljust(p1_cols), "%s" % outside 

    print "\taccesses correlation:".ljust(p1_cols)
    for user_idx,item in enumerate(cluster_agg[w][username][:len(valid_accomplice)]):
        total_access_ratio = sum(map(lambda x: x[1], item['accesses']))
        print ("\t\t%d: accomplice_name:[%s(%s)]" % (user_idx+1, item['name'], get_title(item['name']))).ljust(p3_cols), " simlarity:[%.3e] total_access_ratio:[%.3e]" % (item['prob'], total_access_ratio)
        for item_idx, (access_path, access_ratio) in enumerate(sorted(item['accesses'], key=lambda x: x[1], reverse=True)):
            #TODO title
            print ("\t\t\t-[%s]" % (access_path.encode('utf-8'))).ljust(p3_cols), "%.3e" % access_ratio

    print



            



            


