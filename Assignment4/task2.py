from pyspark import SparkContext
import sys
from collections import defaultdict, deque
from itertools import combinations, permutations

# os.environ['PYSPARK_PYTHON'] = '/usr/local/bin/python3.6'
# os.environ['PYSPARK_DRIVER_PYTHON'] = '/usr/local/bin/python3.6'

sc = SparkContext('local[*]')
sc.setLogLevel("ERROR")

input_path = sys.argv[2]
filter_threshold = int(sys.argv[1])
btw_out_path = sys.argv[3]
com_out_path = sys.argv[4]

df = sc.textFile(input_path)
header = df.first()
df = df.filter(lambda x:x!=header).map(lambda x:x.split(",")).map(lambda x:(x[0],x[1])).groupByKey().mapValues(set).collectAsMap()

# create an empty DataFrame with the specified schema
p = set()
for user1,user2 in combinations(df.keys(),2):
    
    if len(df[user1].intersection(df[user2])) >= filter_threshold:  
            p.add((user1,user2))
            p.add((user2,user1))

edges = sc.parallelize(p)
graph = edges.groupByKey().mapValues(lambda x:set(x)).collectAsMap()
vertices = graph.keys()


def get_communities(updated_graph, vertices):
    
    communities = set()
    #Start vertice - perform BFS to find community
    for start in vertices:
        visited = set()
        queue = deque([start])
        visited.add(start)
        queue = deque([start])
        while queue:
          node = queue.popleft()
          for neighbor in updated_graph[node]:
              if neighbor not in visited:
                 visited.add(neighbor)
                 queue.append(neighbor)

        communities.add(frozenset(visited))

    return communities


def bfs(graph, start):

    num_shortest_paths = {start: 1}
    level = {}
    parent = {}
  
    visited = set() 
    queue = deque([start]) 
    visited.add(start) 
    
    l = 0 
    level[start] = 0
    
    while queue:
        node = queue.popleft() 

        for neighbor in graph[node]:

            if neighbor not in visited:
                visited.add(neighbor) 
                queue.append(neighbor) 
                level[neighbor] = level[node]+1
            
            if  level[neighbor] > level[node]:
                
                v = parent.get(neighbor,[])
                if v==[]:
                  parent[neighbor] = set()
                parent[neighbor].add(node)
                num_shortest_paths[neighbor] = num_shortest_paths.get(neighbor, 0) + num_shortest_paths[node]
                
    return parent,level,num_shortest_paths


def calculate_btw(graph,vertices):

    dicts = []
    for root in vertices:
          c,l,n = bfs(graph,root)
          btw = {}
          btw_path = {}
          d = sorted(l.items(),key=lambda x:-x[1])
          for i,j in d:
              btw[i] = 1
              
          for i,j in d:
              try:
                total = 0
                for prnt in c[i]:
                    total+=n[prnt]

                for prnt in c[i]:
                    v = btw[i]*(n[prnt]/total)
                    btw[prnt] += v
                    # btw[prnt] += v  
                    btw_path[tuple(sorted([i,prnt]))] = v
              except:
                pass
          
          dicts.append(btw_path)

    result = defaultdict(list)
    for d in dicts:
        for key, value in d.items():
            result[key].append(value)

    average = {key: sum(values)/2 for key, values in result.items()}
    average = sorted(average.items(),key=lambda x:(-x[1],x[0][0],x[0][1]))
    return average

average = calculate_btw(graph,vertices)
with open(btw_out_path,"w") as f:
    for k,v in average:
        f.write(str(k)+","+str(round(v,5))+"\n")

degree = {k:len(v) for k,v in graph.items()}
m = len(p)//2

best_q = float('-inf')
while(average):

    mval = average[0][1]
    c = 0
    for i in average:
      if i[1]==mval:
        c+=1

    for i in range(c):
      average.pop(0)

    p = []
    for edge,btw in average:
        p.append((edge[0],edge[1]))
        p.append((edge[1],edge[0]))

    edges = sc.parallelize(p)
    new_graph = edges.groupByKey().mapValues(lambda x:set(x)).collectAsMap()
    for vtx in vertices:
        if vtx not in new_graph.keys():
           new_graph[vtx] = []
          
    new_vertices = new_graph.keys()
    communities = get_communities(new_graph,vertices)
   
    p = set(p)
    q = 0 
    for each in communities:
        
        a = set(permutations(each,2))
        yes_edge = a.intersection(p) 
        no_edge = a.difference(p) 

        for e in yes_edge:
            q += 1 - ((degree[e[0]]*degree[e[1]])/(2*m))
        
        for e in no_edge:
            q += - ((degree[e[0]]*degree[e[1]])/(2*m))

    average = calculate_btw(new_graph,new_vertices)

    if q > best_q:
       best_q = q
       final_communities = communities

    
with open(com_out_path,"w") as f:
     for community in sorted(final_communities, key = lambda x:(len(x),sorted(x)[0])):
         f.write(', '.join(["'"+val+"'" for val in sorted(community)]))
         f.write("\n")




















"""
from collections import defaultdict
dicts = []

for root in vertices:
      c,l,n = bfs(graph,root)
      btw = {}
      btw_path = {}
      d = sorted(l.items(),key=lambda x:-x[1])
      for i,j in d:
          btw[i] = 1
      for i,j in d:
          try:
         
            total = 0
            for prnt in c[i]:
                total+=n[prnt]

            for prnt in c[i]:
                v = btw[i]*(n[prnt]/total)
                btw[prnt] += v
                btw_path[tuple(sorted([i,prnt]))] = v
          except:
            pass
      
      dicts.append(btw_path)

result = defaultdict(list)
for d in dicts:
    for key, value in d.items():
        result[key].append(value)

average = {key: sum(values)/2 for key, values in result.items()}
average = sorted(average.items(),key=lambda x:(-x[1],x[0][0]))
with open(btw_out_path,"w") as f:
     for k,v in average:
         f.write(str(k)+","+str(round(v,5))+"\n")
"""

