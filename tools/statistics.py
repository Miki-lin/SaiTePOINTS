# this file computes statistics of the datasets

import os
import json


stat = {}
def stat_scene(scene):
    
    label_folder = os.path.join(scene)
    label_files  = os.listdir(label_folder)
    
    for file in label_files:
        if os.path.splitext(file)[1] != '.json':
            continue

        with open(os.path.join(label_folder, file)) as f:
            labels = json.load(f)
        
        for l in labels:
            #color = get_color(l["obj_id"])
            obj_type = l["obj_type"]
            if stat.get(obj_type):
                stat[obj_type] = stat[obj_type] + 1
            else:
                stat[obj_type] = 1
            if l['obj_type'] == 'Cart':
                print(file)

    return stat

if __name__=="__main__":
    # for s in os.listdir("../data/haizhu_354"):
    #     print("stat {}".format(s))
    #     stat = stat_scene(s)
    s = '../data/haizhu_1134/label'
    stat = stat_scene(s)
    for x in stat:
        print(x, stat[x])