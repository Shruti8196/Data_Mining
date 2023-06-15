from pyspark import SparkContext
import os

os.environ['PYSPARK_PYTHON'] = '/usr/local/bin/python3.6'
os.environ['PYSPARK_DRIVER_PYTHON'] = '/usr/local/bin/python3.6'
sc = SparkContext('local[*]','wordCount')
rdd = sc.textFile('text.txt')
counts = rdd.flatMap(lambda line:line.split(' ')).map(lambda word:(word,1)).reduceByKey(lambda a,b :a+b).collect()

for each_word in counts:
    print(each_word)
    
#export PYSPARK_PYTHON=python3.6
#export JAVA_HOME=/usr/lib/jvm/java-1.8.0-openjdk-amd64
#/opt/spark/spark-3.1.2-bin-hadoop3.2/bin/spark-submit-executor-memory 4G -driver-memory 4G <sample.py>
