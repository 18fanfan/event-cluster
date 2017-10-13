import os, json
base = "cache/data_transformation/"

is_filter = False

white_list = [
    "EventAnalysis_2016-12-01_2016-12-02"
]

if is_filter:
    file_list = [filename for filename in os.listdir(base) if filename in white_list]
else:
    file_list = os.listdir(base)

for filename in file_list:

    index = json.load(open(base + filename, 'r'))
    docs = index["doc"]
    is_dup = False
    print "checking: ", filename

    doc_id_list = docs.keys()

    for i in range(len(doc_id_list)):
        for j in range(i+1, len(doc_id_list)):
            if docs[doc_id_list[i]] == docs[doc_id_list[j]]:
                print filename , "content duplication"
                print doc_id_list[i], docs[doc_id_list[i]]
                print doc_id_list[j], docs[doc_id_list[j]]
                print 
                is_dup = True
                break

        if is_dup:
            break


