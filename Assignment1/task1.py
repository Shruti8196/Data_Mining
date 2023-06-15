from pyspark import SparkContext
import os
import sys
import json

# os.environ['PYSPARK_PYTHON'] = '/usr/local/bin/python3.6'
# os.environ['PYSPARK_DRIVER_PYTHON'] = '/usr/local/bin/python3.6'
sc = SparkContext('local[*]')
sc.setLogLevel("ERROR")

reviews = sc.textFile(sys.argv[1]).map(json.loads)

d = {}
d["n_review"] = reviews.count()
d["n_review_2018"] = reviews.filter(lambda x:x['date'][:4]=="2018").count()
d["n_user"] = reviews.map(lambda x:(x['user_id'],1)).reduceByKey(lambda a,b:1).count()
d["top10_user"] = reviews.map(lambda x:(x['user_id'],1)).reduceByKey(lambda a,b:a+b).sortBy(lambda x:(-x[1],x[0])).take(10)
d["n_business"] = reviews.map(lambda x:(x['business_id'],1)).reduceByKey(lambda a,b:1).count()
d["top10_business"] = reviews.map(lambda x:(x['business_id'],1)).reduceByKey(lambda a,b:a+b).sortBy(lambda x:(-x[1],x[0])).take(10)

with open(sys.argv[2], 'w') as f:
    json.dump(d,f)
    
#export PYSPARK_PYTHON=python3.6
#export JAVA_HOME=/usr/lib/jvm/java-1.8.0-openjdk-amd64
#/opt/spark/spark-3.1.2-bin-hadoop3.2/bin/spark-submit-executor-memory 4G -driver-memory 4G <sample.py>

#/opt/spark/spark-3.1.2-bin-hadoop3.2/bin/spark-submit --executor-memory 4G --driver-memory 4G
#task1.py <review_filepath> <output_filepath>