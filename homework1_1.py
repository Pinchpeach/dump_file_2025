import matplotlib.pyplot as plt

def Sch_algorithm(A, B, C, X):
  Q = C // A
  R = C % A

  X_n = A * (X % Q) - R * (X // Q) + B % C
  if X_n < 0:
    X_n += C
  return X_n

c = 2**31 - 1
a = 38726
b = 3900
x_0 = 4998

x_result = [x_0]

for i in range(1, 10**5):
  x_next = Sch_algorithm(a, b, c, x_result[i - 1])
  x_result.append(x_next)

for _ in range(10**5):
  x_result[_] = x_result[_] / c

plt.figure()
bin_counts, bin_edges, _ = plt.hist(x_result, bins=50, edgecolor='black', range=(0, 1))
plt.title('Histogram of result')
plt.xlabel('Values')
plt.ylabel('GaeSu')
plt.show()
