from pyspark import SparkContext
import sys,math,time
from itertools import combinations
from collections import defaultdict

# os.environ['PYSPARK_PYTHON'] = '/usr/local/bin/python3.6'
# os.environ['PYSPARK_DRIVER_PYTHON'] = '/usr/local/bin/python3.6'
sc = SparkContext('local[*]')
sc.setLogLevel("ERROR")

def apriori(baskets, support):
    
    c1 = set()
    for b in baskets:
        for item in b:
            c1.add(frozenset({item}))
    
    result = {}
    l1 = check_support(baskets, c1, support)
    curr_L = l1
    k = 2
    
    #continue until all freq itemsets are found
    while (curr_L):
        result[k - 1] = curr_L

        #Generate candidates
        C_current = set()
        for i in curr_L:
            for j in curr_L:
                tmp = i.union(j)
                if (len(tmp) == k):
                    C_current.add(tmp)

        result_pruned = C_current.copy()
        for item in C_current:
            subsets = combinations(item, k-1)
            for s in subsets:
                if (frozenset(s) not in curr_L):
                    result_pruned.remove(item)
                    break
        
        C_current = result_pruned
        curr_L = check_support(baskets, C_current, support)
        k += 1

    return result

def check_support(baskets,candidates, support):
    result = set()
    if len(candidates) == 0:
        return result
    
    counts = defaultdict(int)
    for basket in baskets:
        for candidate in candidates:
            if candidate.issubset(basket):
                counts[candidate] += 1
    
    for itemset, count in counts.items():
        if count >= support:
            result.add(itemset)

    return result

# itemsets is a list of itemsets
def write(itemsets,f):
    c_str = ""
    cline = 1
    for i in sorted(itemsets, key=lambda x: (len(x),sorted(x))):
      t = tuple(sorted(i))
      l = len(t)
      s = str(t)
      if s[-2]==",":
        s = s[:-2]+s[-1]
      if l>cline:
        c_str = c_str[:-1]+"\n\n"
        cline = l 
      c_str+=s
      c_str+=","
    
    f.write(c_str[:-1])
    
    
case_n = int(sys.argv[1])
support = int(sys.argv[2])
input_path = sys.argv[3]
output_path = sys.argv[4]

start = time.time()
rdd = sc.textFile(input_path)
header = rdd.first()
rdd = rdd.filter(lambda x: x!=header).map(lambda x: x.split(','))

if case_n==1:
    rdd = rdd.map(lambda x: [x[0], x[1]])
else:
    rdd = rdd.map(lambda x: [x[1], x[0]])

data = rdd.groupByKey().mapValues(set).map(lambda x: x[1])
n_baskets = data.count()

def Phase1(iterator):
    chunk = list(iterator)
    subset_support = math.ceil((len(chunk) / n_baskets) * support)
    result = apriori(chunk, subset_support)
    frequent_itemsets = set()
    for size, itemsets in result.items():
        for itemset in itemsets:
            frequent_itemsets.add(itemset)
    
    return frequent_itemsets

def Phase2(iterator):
    counts = defaultdict(int)
    chunk = list(iterator)
    for basket in chunk:
        for candidate in candidates:
            if candidate.issubset(basket) :
                counts[candidate] += 1
    return [(itemset, count) for itemset, count in counts.items()]

# Phase1 SON Algorithm
candidates = data.mapPartitions(Phase1).distinct().collect()

with open(output_path, 'w') as f:
    f.write("Candidates:\n")
    write(candidates,f)

# Phase2 SON Algorithm
rdd_out = data.mapPartitions(Phase2).reduceByKey(lambda x,y:x+y).filter(lambda x: x[1] >= support).map(lambda x: x[0])

result = rdd_out.collect()

with open(output_path,'a') as f:
    f.write("\n\nFrequent Itemsets:\n")
    write(result,f)

print("Duration: {0:.2f}".format(time.time() - start))