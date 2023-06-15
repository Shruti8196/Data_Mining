import math
import sys
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score
import numpy as np
import time

        
def intermediate(nth_round,fout):
    ds_point_cnt = 0
    cs_cluster_cnt = 0
    cs_point_cnt = 0
    for key in ds_set.keys():
        ds_point_cnt += ds_set[key][1]
    for key in cs_set.keys():
        cs_cluster_cnt += 1
        cs_point_cnt += cs_set[key][1]
    rs_point_cnt = len(rs_points)
    fout.write(f"{nth_round+1},{ds_point_cnt},{cs_cluster_cnt},{cs_point_cnt},{rs_point_cnt}\n")
    
def get_clusters(labels):
    cluster_dict = {}
    for i, clusterid in enumerate(labels):
        if clusterid in cluster_dict:
            cluster_dict[clusterid].append(i)
        else:
            cluster_dict[clusterid] = [i]
    return cluster_dict

def get_ds_set(c_id, point_indices, data_array):
    ds_set[c_id] = {}
    ds_set[c_id][0] = []
    
    for i in point_indices:
        ds_set[c_id][0].append(true_idx_map[i]) 
        
    n = len(ds_set[key][0])
    sumv = data_array[point_indices, :].astype('float').sum(axis=0)
    sumSq = (data_array[point_indices, :].astype('float')**2).sum(axis=0)
    var = (sumSq/n) - (sumv/n)**2
    
    ds_set[c_id][1] = n
    ds_set[c_id][2] = sumv
    ds_set[c_id][3] = sumSq
    ds_set[c_id][4] = np.sqrt(var)
    ds_set[c_id][5] = sumv/n

def get_cs_set(c_id, point_indices, data_array):
    cs_set[c_id] = {}
    cs_set[c_id][0] = []
    
    global rs_dict, rs_points
    for i in point_indices:
        pointid = list(rs_dict.keys())[list(rs_dict.values()).index(rs_points[i])]
        cs_set[key][0].append(pointid)
        
    n = len(cs_set[key][0])
    sumv = data_array[point_indices, :].astype('float').sum(axis=0)
    sumSq = (data_array[point_indices, :].astype('float')**2).sum(axis=0)
    var = (sumSq/n) - (sumv/n)**2
    
    cs_set[c_id][1] = n
    cs_set[c_id][2] = sumv
    cs_set[c_id][3] = sumSq
    cs_set[c_id][4] = np.sqrt(var)
    cs_set[c_id][5] = sumv/n
    
def get_neighbor(point, stats):   
    #Return -1 if no neighbor found
    nearest_cluster_id = -1
    d = len(point)
    for key, cluster_stats in stats.items():
        std = np.array(cluster_stats[4], dtype=np.float64)
        centroid = np.array(cluster_stats[5], dtype=np.float64)
        mahalanobis_distance = np.sqrt(np.sum(((point - centroid) / std) ** 2))
        if mahalanobis_distance < threshold:
            nearest_cluster_md = mahalanobis_distance
            nearest_cluster_id = key
    return nearest_cluster_id

def get_pcid_dict(ds_stats, cs_stats, rs):
    point_clusterid_map = {}
    for key in ds_stats:
        for point in ds_stats[key][0]:
            point_clusterid_map[point] = key
    for key in cs_stats:
        for point in cs_stats[key][0]:
            point_clusterid_map[point] = -1
    for point in rs:
        point_clusterid_map[point] = -1
    return point_clusterid_map


def write_clustering_results(fout, ds_stats, cs_stats, rs):
    fout.write("\nThe clustering results: ")
    point_clusterid_map = get_pcid_dict(ds_stats, cs_stats, rs)
    for point in sorted(point_clusterid_map.keys(), key=int):
        fout.write("\n" + str(point) + "," + str(point_clusterid_map[point]))

def update_stats(stats,index,add_point,cluster_idx):
    stats[cluster_idx][0].append(index)
    stats[cluster_idx][1] = stats[cluster_idx][1] + 1
    for i in range(0, d):
        stats[cluster_idx][2][i] += add_point[i]
        stats[cluster_idx][3][i] += add_point[i] ** 2
    stats[cluster_idx][4] = np.sqrt((stats[cluster_idx][3][:] / stats[cluster_idx][1]) - (
            np.square(stats[cluster_idx][2][:]) / (stats[cluster_idx][1] ** 2)))
    stats[cluster_idx][5] = stats[cluster_idx][2] / stats[cluster_idx][1]

def merge_cs(cs1_id, cs2_id):
    cs_set[cs1_id][0].extend(cs_set[cs2_id][0])
    cs_set[cs1_id][1] = cs_set[cs1_id][1] + cs_set[cs2_id][1]
    for i in range(0, d):
        cs_set[cs1_id][2][i] += cs_set[cs2_id][2][i]
        cs_set[cs1_id][3][i] += cs_set[cs2_id][3][i]
    cs_set[cs1_id][4] = np.sqrt((cs_set[cs1_id][3][:] / cs_set[cs1_id][1]) - (
            np.square(cs_set[cs1_id][2][:]) / (cs_set[cs1_id][1] ** 2)))
    cs_set[cs1_id][5] = cs_set[cs1_id][2] / cs_set[cs1_id][1]


def merge_cs_ds(cs_id, ds_id):
    ds_set[ds_id][0].extend(cs_set[cs_id][0])
    ds_set[ds_id][1] = ds_set[ds_id][1] + cs_set[cs_id][1]
    for i in range(0, d):
        ds_set[ds_id][2][i] += cs_set[cs_id][2][i]
        ds_set[ds_id][3][i] += cs_set[cs_id][3][i]
    ds_set[ds_id][4] = np.sqrt((ds_set[ds_id][3][:] / ds_set[ds_id][1]) - (
            np.square(ds_set[ds_id][2][:]) / (ds_set[ds_id][1] ** 2)))
    ds_set[ds_id][5] = ds_set[ds_id][2] / ds_set[ds_id][1]

def get_nearest_cluster_map(stats1, stats2):
    cluster1_keys = stats1.keys()
    cluster2_keys = stats2.keys()
    cluster_to_nearest_neighbor = {}
    for key1 in cluster1_keys:
        nearest_cluster_md = threshold
        nearest_clusterid = key1
        for key2 in cluster2_keys:
            if key1 != key2:
                stddev1 = stats1[key1][4]
                centroid1 = stats1[key1][5]
                stddev2 = stats2[key2][4]
                centroid2 = stats2[key2][5]
                md1 = 0
                md2 = 0
                for dim in range(0, d):
                    if stddev2[dim] != 0 and stddev1[dim] != 0:
                        md1 += ((centroid1[dim] - centroid2[dim]) / stddev2[dim]) ** 2
                        md2 += ((centroid2[dim] - centroid1[dim]) / stddev1[dim]) ** 2
                mahalanobis_distance = min(np.sqrt(md1), np.sqrt(md2))
                if mahalanobis_distance < nearest_cluster_md:
                    nearest_cluster_md = mahalanobis_distance
                    nearest_clusterid = key2
        cluster_to_nearest_neighbor[key1] = nearest_clusterid
    return cluster_to_nearest_neighbor

input_file = sys.argv[1]
n_clusters = int(sys.argv[2])
output_file = sys.argv[3]

with open(input_file, "r") as f:
    data = np.array(f.readlines())
    
fout = open(output_file, "w")

# Step 1. Load 20% of the data.
len_20 = int(len(data) * 0.2)
start = 0
end = len_20
initial_load = data[start:end]

first_load, true_idx_map,point_idx_map =[],{},{}
idx = 0
for line in initial_load:
    line = line.split(",")
    index = line[0]
    point = line[2:]
    true_idx_map[idx] = index
    point_idx_map[str(point)] = index
    first_load.append(point)
    idx+=1

d = len(first_load[0])
threshold = 2 * math.sqrt(d)
data_array = np.array(first_load)
model = KMeans(n_clusters=5 * n_clusters, random_state=0)
model.fit(data_array)
labels = model.labels_
clusters = {}
for point, clusterid in zip(first_load,labels):
    if clusterid in clusters:
       clusters[clusterid].append(point)
    else:
       clusters[clusterid] = [point]
rs_dict = {}
for key,val in clusters.items(): 
    if len(val) == 1:
        point = val[0]
        p = first_load.index(point)
        rs_dict[true_idx_map[p]] = point
        first_load.remove(point)
        for l in range(p, len(true_idx_map) - 1):
            true_idx_map[l] = true_idx_map[l + 1]
       
data_minus_rs = np.array(first_load)
model = KMeans(n_clusters=n_clusters, random_state=0)
model.fit(data_minus_rs)
clusters = get_clusters(model.labels_)
ds_set = {}
for key in clusters.keys():
    get_ds_set(key,clusters[key],data_minus_rs)

rs_points = []
for key in rs_dict.keys():
    rs_points.append(rs_dict[key])
rs_data_array = np.array(rs_points)

model = KMeans(n_clusters=int(rs_data_array.shape[0] / 2 + 1), random_state=0)
model.fit(rs_data_array)
cs_clusters = get_clusters(model.labels_)

cs_set = {}
for key,val in cs_clusters.items():
    if len(val) > 1:
       get_cs_set(key,cs_clusters[key],rs_data_array)

for key in cs_clusters.keys():
    if len(cs_clusters[key]) > 1:
        for i in cs_clusters[key]:
            point_to_remove = list(rs_dict.keys())[list(rs_dict.values()).index(rs_points[i])]
            del rs_dict[point_to_remove]

rs_points = []
for key in rs_dict.keys():
    rs_points.append(rs_dict[key])

fout.write("The intermediate results:\n")
intermediate(0,fout)

last_round = 4
for nth_round in range(1, 5):
    start = end
    new_data = []
    if nth_round == last_round:
        end = len(data)
        new_data = data[start:end]
    else:
        end = start + len_20
        new_data = data[start:end]

    points = []
    last_ctr = idx
    for line in new_data:
        line = line.split(",")
        index = line[0]
        point = line[2:]
        points.append(point)
        true_idx_map[idx] = index
        point_idx_map[str(point)] = index
        idx = idx + 1

    new_data_array = np.array(points)
    for i, x in enumerate(new_data_array):
        point = x.astype('float')
        index = true_idx_map[last_ctr + i]
        nearest_cid = get_neighbor(point, ds_set)
        if nearest_cid != -1:
            update_stats(ds_set, index, point, nearest_cid)
        else:
            nearest_cid = get_neighbor(point, cs_set)
            if nearest_cid != -1:
                update_stats(cs_set, index, point, nearest_cid)
            else:
                rs_dict[index] = list(x)
                rs_points.append(list(x))
    
    new_data_array = np.array(rs_points)
    model = KMeans(n_clusters=int(len(rs_points) / 2 + 1), random_state=0)
    model.fit(new_data_array)
    cs_clusters = get_clusters(model.labels_)

    for key in cs_clusters.keys():
        if len(cs_clusters[key]) > 1:
            k = 0
            if key in cs_set.keys():
                while k in cs_set:
                    k = k + 1
            else:
                k = key
            get_cs_set(k, cs_clusters[key], new_data_array)

    for key in cs_clusters.keys():
        if len(cs_clusters[key]) > 1:
            for i in cs_clusters[key]:
                point_to_remove = point_idx_map[str(rs_points[i])]
                if point_to_remove in rs_dict.keys():
                    del rs_dict[point_to_remove]

    rs_points = []
    for key in rs_dict.keys():
        rs_points.append(rs_dict[key])

    cs_ids = cs_set.keys()
    closest_cluster_map = get_nearest_cluster_map(cs_set, cs_set)

    for cs_id in closest_cluster_map.keys():
        if cs_id != closest_cluster_map[cs_id] and closest_cluster_map[cs_id] in cs_set.keys() and cs_id in cs_set.keys():
            merge_cs(cs_id, closest_cluster_map[cs_id])
            del cs_set[closest_cluster_map[cs_id]]
            
    if nth_round == last_round:
        closest_cluster_map = get_nearest_cluster_map(cs_set, ds_set)
        for cs_id in closest_cluster_map.keys():
            if closest_cluster_map[cs_id] in ds_set.keys() and cs_id in cs_set.keys():
                merge_cs_ds(cs_id, closest_cluster_map[cs_id])
                del cs_set[cs_id]

    intermediate(nth_round,fout)

write_clustering_results(fout, ds_set, cs_set, rs_dict)
fout.close()

