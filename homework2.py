import matplotlib.pyplot as plt
import numpy as np

# Shuffling 방법을 이용한 난수발생기
Shuf_list = np.zeros((32,), dtype=float)
seed_val = 176

def Sch_algorithm(A, B, C, X):
  Q = C // A
  R = C % A

  X_n = A * (X % Q) - R * (X // Q) + B % C
  if X_n < 0:
    X_n += C
  return X_n

def Shuf(seed):
  global seed_val
  k = seed
  c = 2**31 - 1
  a = 38726
  b = 3900
  for i in range(32):
    Shuf_list[i] = Sch_algorithm(a, b, c, k)
    k = Shuf_list[i]

  for i in range(32):
    Shuf_list[i] = Shuf_list[i] / c

  a1 = Sch_algorithm(a, b, c, k)
  b1 = Sch_algorithm(a, b, c, a1)

  index = b1 * (10**5) % 32
  
  seed_val = b1

  return Shuf_list[int(index)]

def myran(n=None, seed=None):
  global seed_val
  global Shuf_list
  data_list = []
  
  if n is not None:
    if n == -1:
      for _ in range(32):
        Shuf_list[_] = 0
      return 0
    else:
      for i in range(n):
        value = Shuf(seed_val)
        data_list.append(value)
      return data_list
    
  if seed is not None:
    seed_val = seed
    return 0
  
  return Shuf(seed_val)