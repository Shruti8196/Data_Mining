'''
Method Description:

I used XGBoost as my model.
Added many features from the following files:

business.json: 

1. 'business_id': the unique identifier for the business
2. 'is_open': a binary value indicating whether the business is currently open or closed
3. 'categories': the number of categories the business has
4. 'latitude': the latitude of the business location
5. 'longitude': the longitude of the business location
6. 'BusinessAcceptsCreditCards': a binary value indicating whether the business accepts credit cards
7. 'RestaurantsTakeOut': a binary value indicating whether the business offers takeout service
8. 'GoodForKids': a binary value indicating whether the business is child-friendly
9. 'RestaurantsGoodForGroups': a binary value indicating whether the business is suitable for groups
10. 'RestaurantsPriceRange2': the price range of the business, expressed as a number 
11. 'BikeParking': a binary value indicating whether the business offers bike parking
12. 'ByAppointmentOnly': a binary value indicating whether the business operates by appointment only
13. 'Open24Hours': a binary value indicating whether the business is open 24 hours
14. 'GoodForMeal': lunch: a binary value indicating whether the business is suitable for lunch
15. 'Noise_Level': a numerical value representing the noise level of the business, ranging from 0 to 4
16.  btip.get(x['business_id'],0): a numerical value representing the likes for the business (0 if not available)
17. 'State': a numerical value representing the label assigned to the state where the business is located
18. 'Weekday_hours': the number of hours the business is open on weekdays
19. 'Weekend_hours': the number of hours the business is open on weekends
20. 'Ambience': the number of different types of ambience offered by the business

tip.json: Used to get the total likes for each user and business, the result was augmented to business and user rdd respectively.

user.json:

1.'user_id': the unique identifier for the user
2.'review_count': the number of reviews the user has written
3.'average_stars': the average rating the user has given
4.'yelping_since': the number of days the user has been yelping
5.'useful': the number of times the user's reviews have been marked as useful by other users
6.'funny': the number of times the user's reviews have been marked as funny by other users
7. utip.get(x['user_id'],0): the total tip amount the user has given to businesses 


Error Distribution:
>=0 and <1: 102097
>=1 and <2: 32979
>=2 and <3: 6191
>=3 and <4: 777
>=4: 0

RMSE:
0.979765395

Execution Time:
183.7512ms
'''



from pyspark import SparkContext
import sys,time,json
import xgboost as xgb
from math import sqrt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import math, ast
from datetime import datetime, timedelta
from sklearn.impute import SimpleImputer
import os
from collections import defaultdict

sc = SparkContext('local[*]')
sc.setLogLevel("ERROR")
    
pred_map = {}
model_based = 1
folder_path = sys.argv[1]
test_file_path = sys.argv[2]

        
start = time.time()
def check(s):
    '''Returns the number of categories a business falls into'''
    if s is None:
        return 0
    else:
        return len(s.split(','))

def isopen(s):
    '''Returns 1 if business is open else 0'''
    if s=='True':
        return 1
    else:
        return 0

def attr_bool(attr:dict,key): 
    '''Returns boolean values for attr[key] if exists else 0'''
    if attr is None:
        return 0

    b = attr.get(key,'False')
    if b=='True':
        return 1
    else:
        return 0

def noise_level(attr):
    '''Returns noise levels from (1 to 4) if not null, else returns placeholder -999. '''
    if attr is None:
        return np.nan

    v = attr.get('NoiseLevel',0)
    if v=="quiet":
        return 1
    elif v=="average":
        return 2
    elif v=="loud":
        return 3
    elif v=="very_loud":
        return 4
    else:
        return -999

def goodFor(attr,cat,subcat):
    '''Returns boolean value for goodFor[category].'''
    if attr is None:
        return 0

    b = attr.get(cat,0)
    if b:
        b = ast.literal_eval(b)
        v = b.get(subcat,0)
        if v:
            return 1
        else:
            return 0

def yelping_for(date:str):
    '''Returns the number of days a user is yelping since'''
    date = datetime.strptime(date, "%Y-%m-%d").date()
    current_date = datetime.now().date()
    return (current_date - date).days

def attr_float(attr,key):
    ''' Reurns value for attr[key]'''
    if attr is None:
        return 0

    b = attr.get(key,0)
    return float(b)

def get_ambience_cnt(attr,key):
    ''' Returns the number of ambience settings that matches the business.'''
    if attr is None:
        return 0

    elif attr.get(key,-1)!=-1:
        b = ast.literal_eval(attr[key])
        y = {v for v in b.values() if v}            
        return len(y)
    else:
        return 0
        
attire_map = {}
alabel = 0
def acat_label(attr,attire):
    '''Label encodes new attires, Returns label for the given attire'''
    if not attr:
        return -1
    global attire_map, alabel

    try:
        al = attire_map.get(attr[attire],-1)
        if al==-1:
            attire_map[attire] = alabel
            alabel+=1

            return attire_map[attire]

    except:
       return -1


state_map = {}
slabel = 0
def scat_label(state):
    '''Label encodes new states, Returns label for the given state'''
    if state == "0":
        return "-1"
    global state_map, slabel  
    sl = state_map.get(state,-1)
    if sl==-1:
        state_map[state] = slabel
        slabel+=1

    return state_map[state]


def calculate_weekday_hours(hours_dict):
    '''Calculates total hours a business operates on weekdays'''
    if hours_dict is None:
       return 0
    total_weekday_hours = timedelta()
    for day, hours in hours_dict.items():
        start_time_str, end_time_str = hours.split("-")
        start_time = datetime.strptime(start_time_str, "%H:%M")
        end_time = datetime.strptime(end_time_str.strip(), "%H:%M")

        if day in ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday"):
            total_weekday_hours += (end_time - start_time)

    return total_weekday_hours.total_seconds()/3600

def calculate_weekend_hours(hours_dict):
    '''Calculates total hours a business operates on weekends'''
    if hours_dict is None:
       return 0
    total_weekend_hours = timedelta()
    for day, hours in hours_dict.items():
        start_time_str, end_time_str = hours.split("-")
        start_time = datetime.strptime(start_time_str, "%H:%M")
        end_time = datetime.strptime(end_time_str.strip(), "%H:%M")

        if day in ("Saturday", "Sunday"):
            total_weekend_hours += end_time - start_time

    return total_weekend_hours.total_seconds()/3600


#Reading the tip.json file
tip = sc.textFile(f'{folder_path}/tip.json')
header = tip.first()    
tip = tip.filter(lambda x:x!=header).map(lambda x:json.loads(x)).map(lambda x:(x['user_id'],x['business_id'],x['likes'])).cache()

#utip map: (user_id: total_likes)
utip = tip.map(lambda x:(x[0],x[2])).reduceByKey(lambda x,y:x+y).collectAsMap()

#btip map: (business_id: total_likes)
btip = tip.map(lambda x:(x[1],x[2])).reduceByKey(lambda x,y:x+y).collectAsMap()


#Reading business.json file
business = sc.textFile(f'{folder_path}/business.json')
header = business.first()
#----------------------------------------------------------------------------
business = business.filter(lambda x:x!=header).map(lambda x:json.loads(x)).map(lambda x:(x['business_id'],float(x['stars']),isopen(x.get('is_open',None)),check(x['categories']),x['latitude'],x['longitude'],attr_bool(x['attributes'],"BusinessAcceptsCreditCards"),attr_bool(x['attributes'],"RestaurantsTakeOut"), attr_bool(x['attributes'],"GoodForKids"), attr_bool(x['attributes'],"RestaurantsGoodForGroups"),attr_float(x['attributes'],"RestaurantsPriceRange2"),attr_bool(x['attributes'],"BikeParking"),attr_bool(x['attributes'],"ByAppointmentOnly"),attr_bool(x['attributes'],"Open24Hours"),goodFor(x["attributes"],"GoodForMeal","lunch"),noise_level(x['attributes']),btip.get(x['business_id'],0),scat_label(x.get("state","0")),calculate_weekday_hours(x.get("hours",0)),calculate_weekend_hours(x.get("hours",0)),get_ambience_cnt(x['attributes'],"Ambience")))
print(business.take(1))

#Reading user.json file
user = sc.textFile(f'{folder_path}/user.json')
header = user.first()
#***************************************************************
user = user.filter(lambda x:x!=header).map(lambda x:json.loads(x)).map(lambda x:(x['user_id'],x['review_count'],x['average_stars'],yelping_for(x['yelping_since']),x["useful"],x["funny"],utip.get(x['user_id'],0)))


#***************************************************************
# train: (user_id, business_id, stars)
train = sc.textFile(f'{folder_path}/yelp_train.csv')
header = train.first()
train = train.filter(lambda x:x!=header).map(lambda x:x.split(',')).map(lambda x:(x[0],x[1],float(x[2])))

# val: (user_id, business_id)
val = sc.textFile(test_file_path)
header = val.first()
val = val.filter(lambda x:x!=header).map(lambda x:x.split(',')).map(lambda x:(x[0],x[1]))

#t1: (user_id, (business_id,stars))
t1 = train.map(lambda x:(x[0],(x[1],x[2]))).cache()

#v1: (user_id, (business_id))
v1 = val.map(lambda x:(x[0],(x[1]))).cache()

#u1: (user_id, (review_count, average_stars, yelping_since,useful,funny,utip_likes))***************************************************************
u1 = user.map(lambda x:(x[0],(x[1],x[2],x[3],x[4],x[5],x[6]))).cache()

#b1: (business_id, (isopen,stars,num_categories,.................)) ---------------------------------------
b1 = business.map(lambda x:(x[0],(x[1],x[2],x[3],x[4],x[5],x[6],x[7],x[8],x[9],x[10],x[11],x[12],x[13],x[14],x[15],x[16],x[17],x[18],x[19],x[20]))).cache()

#t1 join u1: (user_id, ((business_id,stars),(review_count, average_stars,yelping_since)) 
# (business_id, (user_id, stars, review_count, average_stars))
# trdata: (business_id,((user_id, stars, review_count, average_stars, yelping_since),(isopen,stars,num_categories)) 
#*************************************************************** x[1][1][add]

trdata = t1.leftOuterJoin(u1).map(lambda x:(x[1][0][0],(x[0],x[1][0][1],x[1][1][0],x[1][1][1],x[1][1][2],x[1][1][3],x[1][1][4],x[1][1][5]))).leftOuterJoin(b1).persist()

#*************************************************************** [1][0][add]
#Xtr: (review_count,average_stars,isopen,business_stars,num_categories,lat,long) ---------------------------
Xtr = trdata.map(lambda x:(x[1][0][2],x[1][0][3],x[1][0][4],x[1][0][5],x[1][0][6],x[1][0][7],x[1][1][0],x[1][1][1],x[1][1][2],x[1][1][3],x[1][1][4],x[1][1][5],x[1][1][6],x[1][1][7],x[1][1][8],x[1][1][9],x[1][1][10],x[1][1][11],x[1][1][12],x[1][1][13],x[1][1][14],x[1][1][15],x[1][1][16],x[1][1][17],x[1][1][18])).collect()

#ytr: (user_bid_stars)
ytr = trdata.map(lambda x:x[1][0][1]).collect()

#***************************************************************[1][1][add]
vldata = v1.leftOuterJoin(u1).map(lambda x:(x[1][0],(x[0],x[1][1][0],x[1][1][1],x[1][1][2],x[1][1][3],x[1][1][4],x[1][1][5]))).leftOuterJoin(b1).persist()

Xvl_info_ids = vldata.map(lambda x:(x[1][0][0],x[0],x[1][0][1])).collect()

#***************************************************************[1][0][add]
#--------------------------------------------------------------------------------------------------------------------
Xvl = vldata.map(lambda x:(x[1][0][1],x[1][0][2],x[1][0][3],x[1][0][4],x[1][0][5],x[1][0][6],x[1][1][0],x[1][1][1],x[1][1][2],x[1][1][3],x[1][1][4],x[1][1][5],x[1][1][6],x[1][1][7],x[1][1][8],x[1][1][9],x[1][1][10],x[1][1][11],x[1][1][12],x[1][1][13],x[1][1][14],x[1][1][15],x[1][1][16],x[1][1][17],x[1][1][18])).collect()

Xtrain = np.array(Xtr)
ytrain = np.array(ytr)

Xval_info_ids = np.array(Xvl_info_ids)

Xvl = np.array(Xvl)

from sklearn.preprocessing import StandardScaler
s = StandardScaler()
s.fit(Xtrain)
Xtrain = s.transform(Xtrain)
Xvl = s.transform(Xvl)

model = xgb.XGBRegressor(grow_policy="lossguide",sampling_method="gradient_based",max_depth=4, learning_rate=0.31, n_estimators=279,objective='reg:linear',seed=0)


model.fit(Xtrain,ytrain)

ypred = model.predict(Xvl)


for i,j in zip(ypred,Xval_info_ids):
    val = pred_map.get((j[0],j[1]),-1)
    if val==-1:
        pred_map[(j[0],j[1])] = []
        
        if i == np.nan:
           #If neither business_id nor user_id in training data, there won't be any features generated
           pred_map[(j[0],j[1])].extend([float(j[2]),np.mean(ytr)])
        else:
           pred_map[(j[0],j[1])].extend([float(j[2]),i])
           

hlen = len(pred_map)
i = 0
with open(sys.argv[3],'w') as f:
      for k,v in pred_map.items():
            
            prediction = v[1]
            f.write(",".join((k[0],k[1],str(prediction))))
            
            if i+1<hlen:
               f.write("\n")
            
            i+=1
             
d = {'user_id':[],'business_id':[],'stars':[]}
for k,v in pred_map.items(): 
  prediction = v[1]
  d['user_id'].append(k[0])
  d['business_id'].append(k[1])
  d['stars'].append(prediction)

df = pd.DataFrame(d)
df1 = pd.read_csv((f"{folder_path}/yelp_val.csv"))
x = df1.merge(df,on=['user_id','business_id'])

# x['abs_diff'] = abs(x['stars_x'] - x['stars_y'])
# edist = defaultdict(int)
# for val in x['abs_diff']:
#     val = abs(val)
#     if val>=0 and val<1:
#       edist[">=0 and <1:"]+=1
#     elif val>=1 and val<2:
#       edist[">=1 and <2:"]+=1
#     elif val>=2 and val<3:
#       edist[">=2 and <3:"]+=1
#     elif val>=3 and val<4:
#       edist[">=3 and <4:"]+=1
#     else:
#       edist[">=4"]+=1
    
# print(edist)
print(sqrt(mean_squared_error(x['stars_x'],x['stars_y'])))
print(time.time()-start)