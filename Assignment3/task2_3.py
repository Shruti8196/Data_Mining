from pyspark import SparkContext
import sys,time,json
import xgboost as xgb
from itertools import combinations
from math import sqrt
import numpy as np
import pandas as pd

sc = SparkContext('local[*]')
sc.setLogLevel("ERROR")
    
hybrid = {}
model_based = 1
folder_path = sys.argv[1]
test_file_path = sys.argv[2]

if model_based:
        
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

          val = sc.textFile(test_file_path)
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

          Xvl_info_ids = vldata.map(lambda x:(x[1][0][0],x[0],x[1][0][1])).collect()
          Xvl = vldata.map(lambda x:(x[1][0][1],x[1][0][2],x[1][1][0],x[1][1][1],x[1][1][2])).collect()

          Xtrain = np.array(Xtr)
          ytrain = np.array(ytr)

          Xval_info_ids = np.array(Xvl_info_ids)
          Xvl = np.array(Xvl)

          model = xgb.XGBRegressor(max_depth=2, learning_rate=0.1, n_estimators=300,objective='reg:linear',seed=0)
          model.fit(Xtrain,ytrain)

          ypred = model.predict(Xvl)

          for i,j in zip(ypred,Xval_info_ids):
              val = hybrid.get((j[0],j[1]),-1)
              if val==-1:
                hybrid[(j[0],j[1])] = []
              
              hybrid[(j[0],j[1])].extend([float(j[2]),i])

def get_prediction(userX,itemY,curr_user_info,ratingMap):

    weights = {}
    strong_neighbors = 0

    # We compare only those items with itemY for which userX have rated
    for item_pair in [(itemY,x) for x in curr_user_info.keys()]:
        try:
          a = ratingMap[item_pair[0]].keys() 
          b = ratingMap[item_pair[1]].keys()
        except:
          weights[item_pair[1]] = 0
          continue

        co_rated = a & b
        if len(co_rated)>=50:

            itemA = []
            itemB = []
            for user in co_rated:
                itemA.append(ratingMap[item_pair[0]][user])
                itemB.append(ratingMap[item_pair[1]][user])

            itemA_avg = sum(itemA)/len(itemA)
            itemB_avg = sum(itemB)/len(itemB)


            itemA_normalized = [i - itemA_avg for i in itemA]
            itemB_normalized = [i - itemB_avg for i in itemB]

            numerator = sum([a*b for a,b in zip(itemA_normalized,itemB_normalized)])
            denominator = sqrt(sum([i**2 for i in itemA_normalized]) * sum([i**2 for i in itemB_normalized]))

            if numerator!=0:
                weights[item_pair[1]] = numerator/denominator

            strong_neighbors += 1

        else:
          weights[item_pair[1]] = 0

    if strong_neighbors>=50:
          # Sort the dictionary items by their values in descending order
          sorted_items = sorted(weights.items(), key=lambda x: x[1], reverse=True)

          # Select the top 50 weights
          weights = {k:v for k,v in sorted_items[:50]}
          numP = 0
          for i in curr_user_info.keys() & weights.keys():
              numP += weights[i]*curr_user_info[i]

          denoP = sum([abs(i) for i in weights.values()])
          if numP!=0:
            return numP/denoP,strong_neighbors

    else:
      #Return average prediction for that item
      try:
          l = ratingMap[itemY].values()
          return sum(l)/len(l),strong_neighbors
      except:
          return 0,strong_neighbors

    return 0,strong_neighbors


item_based = 1
if item_based:

        rdd = sc.textFile(f"{folder_path}/yelp_train.csv")
        header = rdd.first()
        data1 = rdd.filter(lambda x:x!=header).map(lambda x:x.split(",")).persist()

        # business: (user,star)
        data = data1.map(lambda x:(x[1],(x[0],float(x[2])))).groupByKey().mapValues(dict)

        # create broadcast variables
        ratingMap_broadcast = sc.broadcast(data.collectAsMap())
        data_users_broadcast = sc.broadcast(data1.map(lambda x:(x[0],(x[1],float(x[2])))).groupByKey().mapValues(dict).collectAsMap())

        val = sc.textFile(test_file_path)
        header = val.first()
        val = val.filter(lambda x:x!=header).map(lambda x:x.split(','))

        def func(i,j,duser,rmap):
            v = duser.get(i,{})
            return get_prediction(i,j,v,rmap)

        x = val.map(lambda x:(x[0],x[1],func(x[0],x[1],data_users_broadcast.value,ratingMap_broadcast.value))).collect()

        for i in x:
            val = hybrid.get((i[0],i[1]),-1)
            if val==-1:
              hybrid[(i[0],i[1])] = [0,0]
            
            hybrid[(i[0],i[1])].extend([i[2][1],i[2][0]])


#prediction = 𝑓𝑖𝑛𝑎𝑙 𝑠𝑐𝑜𝑟𝑒 = α.𝑠𝑐𝑜𝑟𝑒_𝑖𝑡𝑒𝑚_𝑏𝑎𝑠𝑒𝑑 + (1 − α).𝑠𝑐𝑜𝑟𝑒_𝑚𝑜𝑑𝑒𝑙_𝑏𝑎𝑠𝑒𝑑
#  nn = number of strong neighbors. (neighbors are strong if they have atleast 10 co-rated items)
#  nr = number of reviews received for that item
#  α = nn/(nn+nr) (Note: nn and nr are scaled.)

max_nn = 0
min_nn = float('inf')
max_nr = 0
min_nr = float('inf')

for v in hybrid.values():

    if v[2]>max_nn:
       max_nn = v[2]
          
    if v[0]>max_nr:
       max_nr = v[0]

    if v[2]<min_nn:
       min_nn = v[2]
    
    if v[0]<min_nr:
       min_nr = v[0]

with open(sys.argv[3],'w') as f:
    f.write("user_id, business_id, prediction\n")
    for k,v in hybrid.items():
      nn = (v[2] - min_nn)/(max_nn-min_nn)
      nr = (v[0] - min_nr)/(max_nr-min_nr)
      if nn==0:
        alpha = 0
      else:
        alpha = nn/(nn+nr)
      weighted_prediction = alpha*v[3] + (1-alpha)*v[1]
      f.write(",".join((k[0],k[1],str(weighted_prediction))))
      f.write("\n")

#Print RMSE
d = {'user_id':[],'business_id':[],'stars':[]}
for k,v in hybrid.items():
  nn = (v[2] - min_nn)/(max_nn-min_nn)
  nr = (v[0] - min_nr)/(max_nr-min_nr)
  if nn==0:
    alpha = 0
  else:
    alpha = nn/(nn+nr)
    
  weighted_prediction = alpha*v[3] + (1-alpha)*v[1]
  d['user_id'].append(k[0])
  d['business_id'].append(k[1])
  d['stars'].append(weighted_prediction)

df = pd.DataFrame(d)
df1 = pd.read_csv((f"{folder_path}/yelp_val.csv"))
x = df1.merge(df,on=['user_id','business_id'])

from sklearn.metrics import mean_squared_error
print(sqrt(mean_squared_error(x['stars_x'],x['stars_y'])))


print(time.time()-start)