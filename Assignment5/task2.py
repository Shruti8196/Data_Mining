# Flajolet-Martin algorithm

from blackbox import BlackBox
import binascii
import sys,math
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

def is_prime(n):
    """
    Utility function to check if a number is prime.
    """
    if n <= 1:
        return False
    elif n <= 3:
        return True
    elif n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

def find_nearest_prime(n):
    """
    Function to find the nearest prime number greater than or equal to a given number.
    """
    while not is_prime(n):
        n += 1
    return n

p = find_nearest_prime(stream_size)

def myhash1(s):
    return bin(((2*s+3)%p)%69997)

def myhash2(s):
    return bin(((3*s+5)%p)%69997)

def myhash3(s):
    return bin(((s)%p)%69997)

def myhashs(s):
    result = []
    for f in [myhash1,myhash2,myhash3]:
        result.append(f(s))

    return result

from collections import defaultdict
with open(output_path,"w") as fwriter:

        fwriter.write("Time,Ground Truth,Estimation\n")
        nth_stream = 0
        actual_user_count = set()

        estimated_counts = []
        actual_counts = []
        w = 0 
        while nth_stream <= num_asks: 

              stream_users1 = bx.ask(input_path,stream_size)
              stream_users = [int(binascii.hexlify(s.encode('utf8')),16) for s in stream_users1]
              
              trailing_zeros = defaultdict(list)

              longest_for_h1 = 0
              longest_for_h2 = 0
              longest_for_h3 = 0
              for user in stream_users:
                  
                  actual_user_count.add(user)
                  result = myhashs(user)
                  hfunc = 1
                  try:
                    bit1 = result[0][::-1].index('1')
                  except:
                    bit1 = result[0][::-1].index('b')

                  try:
                    bit2 = result[1][::-1].index('1')
                  except:
                    bit2 = result[1][::-1].index('b')

                  try:
                    bit3 = result[2][::-1].index('1')
                  except:
                    bit3 = result[2][::-1].index('b')

                  longest_for_h1 = max(longest_for_h1,bit1)
                  longest_for_h2 = max(longest_for_h2,bit2)
                  longest_for_h3 = max(longest_for_h3,bit3)

              estimated_count_h1 = 2 ** longest_for_h1
              estimated_count_h2 = 2 ** longest_for_h2
              estimated_count_h3 = 2 ** longest_for_h3
            
              fwriter.write(f"{w},{len(set(stream_users1))},{round((estimated_count_h1+estimated_count_h2+estimated_count_h3)/3)}\n")
              w+=1
              nth_stream+=1