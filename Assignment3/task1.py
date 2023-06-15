from pyspark import SparkContext
import sys,time,random
from itertools import combinations

# os.environ['PYSPARK_PYTHON'] = '/usr/local/bin/python3.6'
# os.environ['PYSPARK_DRIVER_PYTHON'] = '/usr/local/bin/python3.6'
sc = SparkContext('local[*]')
sc.setLogLevel("ERROR")

random.seed(0)
start = time.time()

#Task 1
input_path = sys.argv[1]
output_path = sys.argv[2]
rdd = sc.textFile(input_path)
header = rdd.first()
data = rdd.filter(lambda x:x!=header).map(lambda x:x.split(","))

# map each unique ID to an integer
unique_ids = data.map(lambda x:x[0]).distinct().zipWithIndex()
unique_ids_b = data.map(lambda x:x[1]).distinct().zipWithIndex()

# create a dictionary to map IDs to integers
id_to_int = unique_ids.collectAsMap()
int_to_id = {k:v for v,k in id_to_int.items()}

#Distinct no. of Users and Businesses
n_users = unique_ids.count()
n_business = unique_ids_b.count()

# map the original RDD to a new RDD with integer IDs
new_data = data.map(lambda x: (id_to_int[x[0]],x[1])).groupByKey().mapValues(list)

#A prime number larger than no. of rows
P = 11311

#No. of hash functions
K = 34

# Define the collection of hash functions
hash_func_param = []
for i in range(K):
    a = random.randint(1, P-1,)
    b = random.randint(0, P-1)
    hash_func_param.append((a,b))

def apply_hash(key):
  l = []
  for a,b in hash_func_param:
      l.append((a*key+b)%P)
  return l

#Apply hash to each user_id
hash_info = new_data.map(lambda x:(x[0],apply_hash(x[0])))

#Business: list(user_ids)
business_info = data.map(lambda x:(x[1],x[0])).groupByKey().mapValues(list).collectAsMap()

#Computing the minhash matrix
minhash = [{} for j in range(K)]
for row,h_info in zip(new_data.collect(),hash_info.collect()):
   for hfunc,hval in enumerate(h_info[1]):
      for business in row[1]:
          #if no value available for hfunc,business pair then val = curr_hash_value
          curr_val = minhash[hfunc].get(business,-1)
          if curr_val==-1:
             minhash[hfunc][business] = hval
          #if value exists
          else:
             minhash[hfunc][business] = hval if hval<curr_val else curr_val

#Converting minhash sig to RDD
sig_rdd = sc.parallelize(minhash)
result = sig_rdd.map(lambda x: [(k , [x[k]]) for k in x]).reduce(lambda x,y: x+y)
rdd1 = sc.parallelize(result)
rdd2 = rdd1.reduceByKey(lambda x,y : x+y)

#2 Rows per band - 17 bands
bandinfo = rdd2.map(lambda x: [[(x[0],x[1][i]) for i in range(j,j+2)] for j in range(0,34,2)]).collect()

buckets_info = {}
for index in range(17):
    buckets={}
    for bands in bandinfo:
          b_id = bands[index][0][0]
          str1 =''
          for i in bands[index]:
              str1 += ''.join(str(i[1]))
        
          hbin = hash(str1)
          if hbin not in buckets:
              buckets[hbin] = [b_id]
          else:
              buckets[hbin]+= [b_id]
        
    buckets_info[index] = list(buckets.values())
    del buckets 

buckets = sc.parallelize(list(buckets_info.values())).reduce(lambda x,y: x+y)
candidates = sc.parallelize(buckets).filter(lambda x: len(x) > 1 ).map(lambda x: list(set(x))).map(lambda x:  list(combinations(x,2))).reduce(lambda x,y:x+y)

#Confirming the actual similarity of the candidates selected
similarity = {}
def jaccard(x,y):
    s1 = set(x)
    s2 = set(y)
    return ((len(s1&s2)/float(len(s1|s2))))

for pair in candidates:
    a = pair[0]
    b = pair[1]
    jaccard_value = jaccard(business_info[a],business_info[b])
    if(jaccard_value>=0.5):
        if(a<b):
            #Alphabetical sorting
            if (a,b) not in similarity:
                similarity[(a,b)] = jaccard_value        
        else:
            if (b,a) not in similarity:
                similarity[(b,a)] = jaccard_value 

with open(output_path,"w") as f:

    f.write('business_id_1,business_id_2,similarity\n')
    for item in sorted(similarity.items(),key = lambda item:(item[0][0],item[0][1],item[1])):
        f.write(f'{item[0][0]},{item[0][1]},{item[1]}\n')

print('Runtime',time.time()-start)