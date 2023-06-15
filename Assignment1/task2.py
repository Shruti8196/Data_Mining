from pyspark import SparkContext
import os
import sys
import json
import time

os.environ['PYSPARK_PYTHON'] = '/usr/local/bin/python3.6'
os.environ['PYSPARK_DRIVER_PYTHON'] = '/usr/local/bin/python3.6'
sc = SparkContext('local[*]')
sc.setLogLevel("ERROR")

reviews = sc.textFile(sys.argv[1]).map(json.loads).map(lambda x:(x['business_id'],1)).persist()

#Default partitioning
start = time.time()
result_default = reviews.reduceByKey(lambda a,b:a+b).sortBy(lambda x:(-x[1],x[0]))
end = time.time()
time_duration_default = end - start

n = result_default.getNumPartitions()
len_p = result_default.mapPartitions(lambda p:[sum(1 for _ in p)]).collect()
d = {}
d["default"] = {}
d["default"]["n_partition"] = n
d["default"]["n_items"] = len_p
d["default"]["exe_time"] = time_duration_default


#Custom partitioning

start = time.time()
result_custom = reviews.partitionBy(int(sys.argv[3]),lambda k:hash(k)).reduceByKey(lambda a,b:a+b).sortBy(lambda x:(-x[1],x[0]))
end = time.time()
time_duration_custom = end - start

n1 = result_custom.getNumPartitions()
len_p1 = result_custom.mapPartitions(lambda p:[sum(1 for _ in p)]).collect()

d["customized"] = {}
d["customized"]["n_partition"] = n1
d["customized"]["n_items"] = len_p1
d["customized"]["exe_time"] = time_duration_custom

with open(sys.argv[2], 'w') as f:
    json.dump(d,f)
    
#export PYSPARK_PYTHON=python3.6
#export JAVA_HOME=/usr/lib/jvm/java-1.8.0-openjdk-amd64
#/opt/spark/spark-3.1.2-bin-hadoop3.2/bin/spark-submit-executor-memory 4G -driver-memory 4G <sample.py>