from pyspark import SparkContext
import sys,time
import xgboost as xgb

# os.environ['PYSPARK_PYTHON'] = '/usr/local/bin/python3.6'
# os.environ['PYSPARK_DRIVER_PYTHON'] = '/usr/local/bin/python3.6'
sc = SparkContext('local[*]')
sc.setLogLevel("ERROR")

#Task 3
import time
import pandas as pd
import json
import numpy as np

folder_path = sys.argv[1]

start = time.time()
def check(s):
  if s is None:
    return 0
  else:
    return len(s.split(','))

def isopen(s):
  if s=='True':
    return 1
  else:
    return 0

business = sc.textFile(f'{folder_path}/business.json')
header = business.first()
business = business.filter(lambda x:x!=header).map(lambda x:json.loads(x)).map(lambda x:(x['business_id'],float(x['stars']),isopen(x['is_open']),check(x['categories'])))
user = sc.textFile(f'{folder_path}/user.json')
header = user.first()
user = user.filter(lambda x:x!=header).map(lambda x:json.loads(x)).map(lambda x:(x['user_id'],x['review_count'],x['average_stars']))

train = sc.textFile(f'{folder_path}/yelp_train.csv')
header = train.first()
train = train.filter(lambda x:x!=header).map(lambda x:x.split(',')).map(lambda x:(x[0],x[1],float(x[2])))

val = sc.textFile(f'{sys.argv[2]}')
header = val.first()
val = val.filter(lambda x:x!=header).map(lambda x:x.split(',')).map(lambda x:(x[0],x[1]))

t1 = train.map(lambda x:(x[0],(x[1],x[2]))).cache()
v1 = val.map(lambda x:(x[0],(x[1]))).cache()
u1 = user.map(lambda x:(x[0],(x[1],x[2]))).cache()
b1 = business.map(lambda x:(x[0],(x[1],x[2],x[3]))).cache()

trdata = t1.leftOuterJoin(u1).map(lambda x:(x[1][0][0],(x[0],x[1][0][1],x[1][1][0],x[1][1][1]))).leftOuterJoin(b1).persist()

Xtr = trdata.map(lambda x:(x[1][0][2],x[1][0][3],x[1][1][0],x[1][1][1],x[1][1][2])).collect()
ytr = trdata.map(lambda x:x[1][0][1]).collect()

vldata = v1.leftOuterJoin(u1).map(lambda x:(x[1][0],(x[0],x[1][1][0],x[1][1][1]))).leftOuterJoin(b1).persist()

Xvl_info_ids = vldata.map(lambda x:(x[1][0][0],x[0])).collect()
Xvl = vldata.map(lambda x:(x[1][0][1],x[1][0][2],x[1][1][0],x[1][1][1],x[1][1][2])).collect()

Xtrain = np.array(Xtr)
ytrain = np.array(ytr)

Xval_info_ids = np.array(Xvl_info_ids)
Xvl = np.array(Xvl)

model = xgb.XGBRegressor(max_depth=2, learning_rate=0.1, n_estimators=300,objective='reg:linear',seed=0)
model.fit(Xtrain,ytrain)

ypred = model.predict(Xvl)

with open(sys.argv[3],"w") as f:
      f.write('user_id, business_id, prediction\n')
      for i,j in zip(ypred,Xval_info_ids):
          f.write(f"{j[0]},{j[1]},{i}\n")

print(time.time()-start)