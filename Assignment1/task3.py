from pyspark import SparkContext
import os
import sys
import json
import time

os.environ['PYSPARK_PYTHON'] = '/usr/local/bin/python3.6'
os.environ['PYSPARK_DRIVER_PYTHON'] = '/usr/local/bin/python3.6'
sc = SparkContext('local[*]')
sc.setLogLevel("ERROR")

#Task 3
reviews = sc.textFile(sys.argv[1]).map(lambda x: json.loads(x)).persist()
business = sc.textFile(sys.argv[2]).map(lambda x: json.loads(x)).persist()
b = business.map(lambda x:(x['business_id'],x['city']))
r = reviews.map(lambda x:(x['business_id'],x['stars']))
avg = b.join(r)\
.map(lambda x:x[1])\
.mapValues(lambda x: (x, 1))\
.reduceByKey(lambda x, y: (x[0]+y[0], x[1]+y[1]))\
.mapValues(lambda x: x[0]/x[1])

d = {}
start = time.time()
top10_cities = avg.sortBy(lambda x:(-x[1],x[0])).take(10)
d["m1"] = time.time()-start


start = time.time()
top10_cities = sorted(avg.collect(),key=lambda x:(-x[1],x[0]))[:10]
d["m2"] = time.time()-start

d["reason"] = "For small datasets, 'sorted' in Python may be faster, as it sorts the data in-place. But, for large datasets, 'sortBy' in PySpark can be more efficient, as it takes advantage of Spark's distributed processing capabilities to sort the data in parallel across multiple nodes."

with open(sys.argv[3],"w") as f:
    f.write('city,stars')
    for i in avg.sortBy(lambda x:(-x[1],x[0])).collect():
      f.write("\n")
      f.write(f"{i[0]},{i[1]}")
    
with open(sys.argv[4],"w") as f:
    json.dump(d,f)

#export PYSPARK_PYTHON=python3.6
#export JAVA_HOME=/usr/lib/jvm/java-1.8.0-openjdk-amd64
#/opt/spark/spark-3.1.2-bin-hadoop3.2/bin/spark-submit-executor-memory 4G -driver-memory 4G <sample.py>
