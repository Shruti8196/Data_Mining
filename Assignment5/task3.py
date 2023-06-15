# Flajolet-Martin algorithm

from blackbox import BlackBox
import random
import sys
from pyspark import SparkContext
sc = SparkContext('local[*]')
sc.setLogLevel("ERROR")

input_path = sys.argv[1]
stream_size = int(sys.argv[2])
num_asks = int(sys.argv[3])
output_path = sys.argv[4]
bx = BlackBox()
random.seed(553)

with open(output_path,"w") as fwriter:

        fwriter.write("seqnum,0_id,20_id,40_id,60_id,80_id\n")
        nth_stream = 0
        memory = []
        n = 0
        seq = 0
        while nth_stream < num_asks: 
              
            stream_users = bx.ask(input_path,stream_size)
            for user in stream_users:
                n +=1
                if n<=100:
                   memory.append(user)
                else:
                   if(random.random() < (100/n)):
                      memory[random.randint(0,99)] = user

                if n>=100 and n%100==0:
                   fwriter.write(f"{n},{memory[0]},{memory[20]},{memory[40]},{memory[60]},{memory[80]}\n")
            nth_stream+=1