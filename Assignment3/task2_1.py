from pyspark import SparkContext
import sys,time
from itertools import combinations
from math import sqrt

# os.environ['PYSPARK_PYTHON'] = '/usr/local/bin/python3.6'
# os.environ['PYSPARK_DRIVER_PYTHON'] = '/usr/local/bin/python3.6'
sc = SparkContext('local[*]')
sc.setLogLevel("ERROR")

def get_prediction(userX,itemY,curr_user_info,ratingMap):
    weights = {}
    flag = 0

    # We compare only those items with itemY for which userX have rated
    for item_pair in [(itemY,x) for x in curr_user_info.keys()]:
        try:
          a = ratingMap[item_pair[0]].keys() 
          b = ratingMap[item_pair[1]].keys()
        except:
          weights[item_pair[1]] = 0
          continue

        co_rated = a & b
        
        if len(co_rated)>=10:
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

            flag += 1

        else:
          weights[item_pair[1]] = 0

    if flag>=10:
          # Sort the dictionary items by their values in descending order
          sorted_items = sorted(weights.items(), key=lambda x: x[1], reverse=True)
          
          # Select the top 50 weights
          weights = {k:v for k,v in sorted_items[:10]}
          numP = 0
          for i in curr_user_info.keys() & weights.keys():
              numP += weights[i]*curr_user_info[i]

          denoP = sum([abs(i) for i in weights.values()])
          if numP!=0:
            return numP/denoP

    else:
      #Return average prediction for that item
      try:
          l = ratingMap[itemY].values()
          return sum(l)/len(l)
      except:
          return 0

    return 0


rdd = sc.textFile(sys.argv[1])
header = rdd.first()
data1 = rdd.filter(lambda x:x!=header).map(lambda x:x.split(",")).persist()

# business: (user,star)
data = data1.map(lambda x:(x[1],(x[0],float(x[2])))).groupByKey().mapValues(dict)

# create broadcast variables
ratingMap_broadcast = sc.broadcast(data.collectAsMap())
data_users_broadcast = sc.broadcast(data1.map(lambda x:(x[0],(x[1],float(x[2])))).groupByKey().mapValues(dict).collectAsMap())

val_path = sys.argv[2]
val = sc.textFile(val_path).map(lambda x:x.split(','))
header = val.first()
val = val.filter(lambda x:x!=header)

def func(i,j,duser,rmap):
    v = duser.get(i,{})
    return get_prediction(i,j,v,rmap)

x = val.map(lambda x:(x[0],x[1],func(x[0],x[1],data_users_broadcast.value,ratingMap_broadcast.value))).collect()

with open(sys.argv[3],"w") as f:
     f.write('user_id, business_id, prediction\n')
     for i in x:
         f.write(','.join((i[0],i[1],str(i[2]))))
         f.write('\n')