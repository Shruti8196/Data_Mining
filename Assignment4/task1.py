import sys,os
from pyspark.sql.types import StructType, StructField, IntegerType, StringType
from graphframes import *
from itertools import combinations
from pyspark import SparkContext
from pyspark.sql import SparkSession

os.environ["PYSPARK_SUBMIT_ARGS"] = "--packages graphframes:graphframes:0.8.2-spark3.1-s_2.12 pyspark-shell"

spark = SparkSession.builder.master("local[*]").getOrCreate()
sc = spark.sparkContext
input_path = sys.argv[2]
filter_threshold = int(sys.argv[1])
output_path = sys.argv[3]

sc.setLogLevel("ERROR")

df = sc.textFile(input_path)
header = df.first()
df = df.filter(lambda x:x!=header).map(lambda x:x.split(",")).map(lambda x:(x[0],x[1])).groupByKey().mapValues(set).collectAsMap()
schemav = StructType([StructField('user_id',StringType(),True)])

schemap = StructType([
    
    StructField('user_id1',StringType(),True),
    StructField('user_id2',StringType(),True)

])

# create an empty DataFrame with the specified schema
p = set()
v = set()
for user1,user2 in combinations(df.keys(),2):
    
    if len(df[user1].intersection(df[user2])) >= filter_threshold:
             
            p.add((user1,user2,"edge"))
            p.add((user2,user1,"edge"))
            v.add((user1,))
            v.add((user2,))

vertices = spark.createDataFrame(v,["id"])
edges = spark.createDataFrame(p,["src", "dst","relationship"])
g = GraphFrame(vertices, edges)

# run LPA algorithm
result = g.labelPropagation(maxIter=5)

# show the communities
result1 = result.rdd

result1 = result1.map(lambda x:(x[1],x[0])).groupByKey().mapValues(set).cache()

with open(output_path,'w') as f:
    
    for community in sorted(result1.collect(), key=lambda x: (len(x[1]), sorted(x[1])[0])):
        community_str = ', '.join(["'"+v+"'" for v in sorted(community[1])])
        f.write(community_str + "\n")
        
