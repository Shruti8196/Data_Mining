import numpy as np
import sys
from sklearn.cluster import KMeans
from itertools import combinations
input_file = sys.argv[1]
n_cluster = int(sys.argv[2])
output_file = sys.argv[3]

with open(output_file,"w") as f:
     f.write("The intermediate results:\n")
        
#Read input
data = []
with open(input_file, "r") as file:
    for line in file:
        line = line.strip().split(",")
        data.append(line[1:])

data = np.array(data,dtype='float') 
np.random.seed(42)
np.random.shuffle(data)
data_passes = {}
partition = 0
len_20 = int(len(data)*0.2)
j = 0
for i in range(0,len(indices),len_20):
    data_passes[f"data_{partition}"] = {}
    data_passes[f"data_{partition}"]["data"] = data[i:len_20+j,1:]
    data_passes[f"data_{partition}"]["true_labels"] = data[i:len_20+j,0]
    j = j+len_20
    partition+=1
    
#read
data_read = 0
data_20 = data_passes[f"data_{data_read}"]["data"]
n_dim = len(data_20[0])
data_read+=1
k = n_cluster*5

model = KMeans(n_clusters=k,random_state=42)
model.fit(data_20)
labels = model.labels_

#Get labels with count
n_elements_cen = np.unique(labels,return_counts=True)

#Get clusters ids that has only one point
index = np.where(n_elements_cen[1]==1)[0]

rs = np.array([]).reshape(0,n_dim)
rs_indices = []

#For each cluster that has only one element
for each_label in index:
    #Move all the clusters with one point to RS Set
    
    rs_point_index = 0
    for i in labels:
        if i==each_label:
           break
        rs_point_index+=1
              
    #rs_point_index = np.where(labels==each_label)[0]
    rs_indices.append(rs_point_index)
    rs = np.append(rs,data_20[rs_point_index].reshape(1,n_dim),axis=0)

for i in rs_indices:
    data_20 = np.delete(data_20,i,axis=0)
    k-=1
    
# Run K-Means again to cluster the rest of the data points with K = the number of input clusters.
model = KMeans(n_clusters=k,random_state=42)
model.fit(data_20)
labels = model.labels_

#Use the K-Means result from Step 4 to generate the DS clusters (i.e., discard their points and generate statistics)
ds_stats = {}

#Count elements in each cluster
n_elements_cen = np.unique(labels,return_counts=True)

#Iterate over cluster number and cluster N
for label,n in zip(n_elements_cen[0],n_elements_cen[1]):

    if n==0:
       continue
    idx = np.where(labels==label)[0]
    sm = data_20[idx].sum(axis=0).reshape(1,n_dim)
    sumsq = (data_20[idx]**2).sum(axis=0).reshape(1,n_dim)  
    var = ((sumsq/n) - ((sm/n)**2)).reshape(1,n_dim)

    ds_stats[label] = {}
    ds_stats[label]['N'] = n
    ds_stats[label]['sum'] = sm
    ds_stats[label]['sumSq'] = sumsq
    ds_stats[label]['std'] = np.sqrt(var)
    ds_stats[label]['centroid'] = sm/n


# Run K-Means on the points in the RS with a large K (e.g., 5 times of the number of the input 
# clusters) to generate CS (clusters with more than one points) and RS (clusters with only one point).

if rs.shape[0]>=1:
      model = KMeans(n_clusters=(len(rs)//2+1))
      model.fit(rs)
      labels = model.labels_

      #Calculating cs_stats
      cs_stats = {}
      rs_new = np.array([]).reshape(0,n_dim)

      #Count elements in each cluster
      n_elements_cen = np.unique(labels,return_counts=True)
      
      cs_label = 0
      #Iterate over cluster number and cluster N
      for label,n in zip(n_elements_cen[0],n_elements_cen[1]):

          idx = np.where(labels==label)[0]
          if n==0:
              continue
          if n==1:
              #Move all the clusters with one point to RS Set
              rs_new = np.append(rs_new,data_20[idx],axis=0)
          else:
              #If more than one point then create a CS Set
              sm = rs[idx].sum(axis=0).reshape(1,n_dim)
              centroid = (sm/n).reshape(1,n_dim)
              sumsq = (rs[np.where(labels==label)[0]]**2).sum(axis=0).reshape(1,n_dim)
              var = ((sumsq/n) - ((sm/n)**2)).reshape(1,n_dim)

              cs_stats[cs_label] = {}
              cs_stats[cs_label]['N'] = n
              cs_stats[cs_label]['sum'] = sm
              cs_stats[cs_label]['sumSq'] = sumsq
              cs_stats[cs_label]['std'] = np.sqrt(var)
              cs_stats[cs_label]['centroid'] = sm/n
              cs_label+=1

      rs = rs_new

else:
      cs_stats = {}
      cs_label = 0

def intermediate(nth_round,ds_stats,cs_stats,rs):

    total_points_ds = 0
    total_clusters_cs = 0
    total_points_cs = 0
    for k,v in ds_stats.items():
        total_points_ds+=ds_stats[k]["N"]

    for k,v in cs_stats.items():
        total_points_cs+=cs_stats[k]["N"]
        total_clusters_cs+=1

    with open(output_file,"a") as f:
         f.write(f"Round {nth_round}: {total_points_ds},{total_clusters_cs},{total_points_cs},{rs.shape[0]}\n")
   
def merge_clusters(cs_stats,cs_stats_new):
    
    next_label = list(cs_stats.keys())[-1]+1

    l = []

    #Iterate over old clusters
    for key,value in cs_stats.copy().items():
        c = cs_stats[key]["centroid"]
        std = cs_stats[key]["std"]
        if np.any(std)==0:
           std+=1e-10

        #For each new cluster
        for key1,value1 in cs_stats_new.items():
            point = value1["centroid"]
            y = (point - c)/std

            if (np.sqrt((y**2).sum(axis=1))) < 2*np.sqrt(n_dim):
                #Merge clusters 
                cs_stats[key]["N"] += cs_stats_new[key1]["N"]
                cs_stats[key]["sum"]+= cs_stats_new[key1]["sum"]
                cs_stats[key]["sumSq"]+= cs_stats_new[key1]["sumSq"]
                cs_stats[key]["centroid"]=cs_stats[key]["sum"]/cs_stats[key]["N"]

                var = (cs_stats[key]["sumSq"]/cs_stats[key]["N"]) - ((cs_stats[key]["sum"]/cs_stats[key]["N"])**2)
                cs_stats[key]["std"] = np.sqrt(var)

            else:
                l.append(value1)
        
        for i in l:
            cs_stats[next_label] = i
            next_label+=1

    return cs_stats

intermediate(data_read-1,ds_stats,cs_stats,rs)
rs = np.array([]).reshape(0,n_dim)
lp = len(data_passes)
while(data_read<lp):
    
    #Read the next batch
    data_20 = data_passes[f"data_{data_read}"]["data"]
    data_read+=1

    #For each point in the batch do
    for point in data_20:
        
        assigned_ds,assigned_cs = 0,0
        point = point.reshape(1,n_dim)
        for key, value in ds_stats.items():
            
            c = ds_stats[key]["centroid"]
            std = ds_stats[key]["std"]
            if np.any(std)==0:
              #smaller value adjustment
              std+=1e-10

            y = (point - c)/std
            #If condition satisfied, add to ds_set, i.e update ds metadata
            if (np.sqrt((y**2).sum())) < 2*np.sqrt(n_dim):
                  assigned_ds = 1     
                  #update metadata 
                  ds_stats[key]["N"] = ds_stats[key]["N"]+1
                  ds_stats[key]["sum"]+=point
                  ds_stats[key]["sumSq"]+=(point**2)
                  ds_stats[key]["centroid"]=ds_stats[key]["sum"]/ds_stats[key]["N"]

                  var = (ds_stats[key]["sumSq"]/ds_stats[key]["N"]) - ((ds_stats[key]["sum"]/ds_stats[key]["N"])**2)
                  ds_stats[key]["std"] = np.sqrt(var)
                  break

        if assigned_ds==0:
          
            #Assign to cs set if satisfies condition
            for key, value in cs_stats.items():
                c = cs_stats[key]["centroid"]
                std = cs_stats[key]["std"]
                if np.any(std)==0:
                    std+=1e-10

                y = (point - c)/std
                #If condition satisfied, add to cs_set, i.e update cs metadata
                if (np.sqrt((y**2).sum(axis=1))) < 2*np.sqrt(n_dim):
                  
                    assigned_cs = 1
                    #update cs metadata 
                    cs_stats[key]["N"] = cs_stats[key]["N"]+1
                    cs_stats[key]["sum"]+=point
                    cs_stats[key]["sumSq"]+=(point**2)
                    cs_stats[key]["centroid"]=cs_stats[key]["sum"]/cs_stats[key]["N"]

                    var = (cs_stats[key]["sumSq"]/cs_stats[key]["N"]) - ((cs_stats[key]["sum"]/cs_stats[key]["N"])**2)
                    cs_stats[key]["std"] = np.sqrt(var)
                    break
                    
        if assigned_cs==0 and assigned_ds==0:
            rs = np.append(rs,point.reshape(1,n_dim),axis=0)

    if rs.shape[0]>=1:

        model = KMeans(n_clusters=(len(rs)//2)+1)
        model.fit(rs)
        labels = model.labels_

        #Updating cs_stats with new clusters if any
        rs_new = np.array([]).reshape(0,n_dim)

        #Count elements in each cluster
        n_elements_cen = np.unique(labels,return_counts=True)

        cs_label_new = 0
        #Iterate over cluster number and cluster N
        for label,n in zip(n_elements_cen[0],n_elements_cen[1]):

            idx = np.where(labels==label)[0]
            if n==0:
                continue
            if n==1:
                #Move all the clusters with one point to RS Set
                rs_point_index = np.where(labels==label)[0]
                rs_new = np.append(rs_new,rs[rs_point_index],axis=0)
            else:
                #If more than one point then create a CS Set
                cs_stats_new = {}
                cs_stats_new[cs_label_new] = {}

                sm = rs[idx].sum(axis=0).reshape(1,n_dim)
                centroid = (sm/n).reshape(1,n_dim)
                sumsq = (rs[np.where(labels==label)[0]]**2).sum(axis=0).reshape(1,n_dim)
                var = ((sumsq/n) - ((sm/n)**2)).reshape(1,n_dim)
         
                cs_stats_new[cs_label_new]['N'] = n
                cs_stats_new[cs_label_new]['sum'] = sm
                cs_stats_new[cs_label_new]['sumSq'] = sumsq
                cs_stats_new[cs_label_new]['std'] = np.sqrt(var)
                cs_stats_new[cs_label_new]['centroid'] = sm/n
                cs_label_new+=1

 
        if cs_stats:
           cs_stats = merge_clusters(cs_stats,cs_stats_new)
        else:
           cs_stats = cs_stats_new

        rs = rs_new
    
    if data_read==lp:
       continue
    intermediate(data_read-1,ds_stats,cs_stats,rs)

#Last Run: Merge CS to DS
ds_stats = merge_clusters(ds_stats,cs_stats)
intermediate(data_read-1,ds_stats,cs_stats,rs)


def write_clustering_results(ds_stats,cs_stats,rs):
    with open(output_file,"a") as f:
         f.write("\nThe clustering results:\n")
         
         ele_ind = 0
         label_ind = 0
         for k,v in ds_stats.items():
             for dpindex in range(v["N"]):
                  f.write(f"{ele_ind},{label_ind}\n")
                  ele_ind+=1
             
             label_ind+=1

         for k,v in cs_stats.items():
             for dpindex in range(v["N"]):
                  f.write(f"{ele_ind},{label_ind}\n")
                  ele_ind+=1
             
             label_ind+=1

         for ele in rs:
              f.write(f"{ele_ind},-1\n")
              ele_ind+=1
            
            
write_clustering_results(ds_stats,cs_stats,rs)            