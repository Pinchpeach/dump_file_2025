import numpy as np
import matplotlib.pyplot as plt

def given_func(x):
    if (x > 0 and x < np.pi):
        return 1
    elif (x < 0 and x > -np.pi):
        return -1
    else:
        return 0
    
func_vec = np.vectorize(given_func)

gap = np.linspace(-np.pi, np.pi, 1000)
result_list = []
for _ in range(20):
    n = 2 * _ + 1
    result_list.append(_)
    result_list[_] = (4 * np.sin(n * gap) / (n * np.pi))
    
approx = np.zeros(len(gap))

plt.figure()
plt.plot(gap, func_vec(gap), color='black')
for _ in range(20):
    approx += result_list[_]
    plt.plot(gap, approx, alpha=0.5)
    
    
    
