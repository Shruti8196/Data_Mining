from blackbox import BlackBox
import binascii
import sys
from pyspark import SparkContext
sc = SparkContext('local[*]')
sc.setLogLevel("ERROR")

input_path = sys.argv[1]
stream_size = int(sys.argv[2])
num_asks = int(sys.argv[3])
output_path = sys.argv[4]

bx = BlackBox()
filter_bit_arr = [0]*69997
seen = set()

# FNV-1a (Fowler-Noll-Vo) Hash 
def myhash13(x):

    FNV_offset_basis = 14695981039346656037
    FNV_prime = 1099511628211
    
    # Truncate x to 64 bits
    x = x & ((1 << 64) - 1)

    # Convert integer to bytes
    x_bytes = x.to_bytes(8, byteorder='big')
    hash_val = FNV_offset_basis
    for b in x_bytes:
        hash_val ^= b
        hash_val *= FNV_prime
    return hash_val % 69997

def myhash23(x):
    FNV_offset_basis = 2166136261
    FNV_prime = 16777619
    x = x & ((1 << 64) - 1)
    x_bytes = x.to_bytes(8, byteorder='big')
    hash_val = FNV_offset_basis
    for b in x_bytes:
        hash_val ^= b
        hash_val *= FNV_prime
    return hash_val % 69997

def myhash1(s):
    return ((101*s+17)%70001)%69997

def myhash2(s):
    return ((307*s+89)%70003)%69997

def myhashs(s):
    result = []
    for f in [myhash1,myhash2]:
        result.append(f(s))

    return result


with open(output_path,"w") as fwriter:

        fwriter.write("Time,FPR\n")
        time = 0
        fpr = []
        
        for _ in range(num_asks):
            stream_users = bx.ask(input_path,stream_size)
            stream_users = [int(binascii.hexlify(s.encode('utf8')),16) for s in stream_users]
            fp = 0
            tn = 0
            for user in stream_users:
                result = myhashs(user)
                f = 0
                decision = 0 
                for index in result:
                    
                    if filter_bit_arr[index]==0:
                      # Not seen proved
                      f = 1

                    filter_bit_arr[index] = 1

                if f==0: #Element was seen before
                  decision = 1 #Seen

                if decision == 1 and user not in seen:
                  #Increment False positive count by 1
                  fp+=1
                  
                elif decision == 0 and user not in seen:
                  #Increment True Negative count by 1
                  tn+=1
                seen.add(user)
          
            fwriter.write(f"{time},{fp/(fp+tn)}\n")
            time+=1