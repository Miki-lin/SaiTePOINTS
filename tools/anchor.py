# this file computes statistics of the datasets

import os
import json
import numpy as np

stat = {}


def stat_scene(scene):
    label_folder = os.path.join(scene)
    label_files = os.listdir(label_folder)

    for file in label_files:
        if os.path.splitext(file)[1] != '.json':
            continue

        with open(os.path.join(label_folder, file)) as f:
            labels = json.load(f)

        for l in labels:
            # color = get_color(l["obj_id"])
            obj_type = l["obj_type"]
            lwh = list(l['psr']['scale'].values())
            if stat.get(obj_type):
                stat[obj_type].append(lwh)
            else:
                stat[obj_type] = []
                stat[obj_type].append(lwh)

    return stat


def ou_dis(vec1, vec2):
    # dis = np.sqrt(np.square(vec1, vec2))
    vec1 = vec1[:, None]
    vec2 = vec2[None]
    vec3 = np.repeat(vec2, vec1.shape[0], axis=0)
    dis = np.sqrt(np.sum(np.square(vec1 - vec3), axis=2))
    return dis


def k_means(boxes, k, dist=np.median):
    """
    yolo k-means methods
    refer: https://github.com/qqwweee/keras-yolo3/blob/master/kmeans.py
    Args:
        boxes: 需要聚类的bboxes
        k: 簇数(聚成几类)
        dist: 更新簇坐标的方法(默认使用中位数，比均值效果略好)
    """
    box_number = boxes.shape[0]
    last_nearest = np.zeros((box_number,))
    # np.random.seed(0)  # 固定随机数种子

    # init k clusters
    clusters = boxes[np.random.choice(box_number, k, replace=False)]

    while True:
        # clusters_all = np.repeat(clusters, box_number, axis=0)
        distances = ou_dis(boxes, clusters)
        current_nearest = np.argmin(distances, axis=1)
        if (last_nearest == current_nearest).all():
            break  # clusters won't change
        for cluster in range(k):
            # update clusters
            clusters[cluster] = dist(boxes[current_nearest == cluster], axis=0)

        last_nearest = current_nearest

    return clusters


if __name__ == "__main__":
    # for s in os.listdir("../data/haizhu_354"):
    #     print("stat {}".format(s))
    #     stat = stat_scene(s)
    # anchor=1,mean
    # anchor>=2,k_means
    s = '../data/haizhu_1134/label'
    stat = stat_scene(s)
    for cls in stat:
        print(cls)
        whl = np.array(stat[cls])
        clusters = k_means(whl, 2)
        print('k_means:')
        print(clusters)
        means = np.average(whl, axis=0)
        print('means:')
        print(means)
