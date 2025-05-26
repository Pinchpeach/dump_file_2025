import numpy as np
import matplotlib.pyplot as plt

def find_pos(x, xdata):
    cpy_JD = xdata.copy()

    low = 0
    high = len(cpy_JD) - 1

    while high - low > 1:
        mid = (low + high) // 2
        if xdata[mid] > x:
            high = mid
        else:
            low = mid
    
    return low

def cubic_spline(ip_pos, xdata, ydata):
    # initial setting. make matrix for calculation
    n = len(xdata) - 2
    # n defines the size of matrix

    derv_two = np.zeros(n).T

    mat_cal = np.zeros((n, n))

    res_cal = np.zeros(n).T

    for _ in range(n):
        mat_cal[_][_] = (xdata[_ + 2] - xdata[_]) / 3
        if _ != 0:
           mat_cal[_][_ - 1] = (xdata[_ + 1] - xdata[_]) / 6
        if _ != len(xdata) - 3:
            mat_cal[_][_ + 1] = (xdata[_ + 2] - xdata[_ + 1]) / 6
    res_cal[_] = ((ydata[_ + 2] - ydata[_ + 1]) / (xdata[_ + 2] - xdata[_ + 1])) - ((ydata[_ + 1] - ydata[_]) / (xdata[_ + 1] - xdata[_]))

    derv_two = np.linalg.solve(mat_cal, res_cal)

    derv_two = np.append(derv_two, 0)
    derv_two = np.append(0, derv_two)

    # find the position of interpolation
    index = find_pos(ip_pos, xdata)

    x0 = xdata[index]
    x1 = xdata[index + 1]
    
    y0 = ydata[index]
    y1 = ydata[index + 1]

    t = x1 - x0
    A = (x1 - ip_pos) / t
    B = (ip_pos - x0) / t
    C = (A**3 - A) * (t**2) / 6
    D = (B**3 - B) * (t**2) / 6

    return A * y0 + B * y1 + C * derv_two[index] + D * derv_two[index + 1] 

def func_given(t):
    return np.sin(t) + 0.2 * np.cos(3*t)

gap_test = np.linspace(0, 20, 1000)
index_gap = ["100", "20", "10", "4"]
index_sam = ["sam_100", "sam_20", "sam_10", "sam_4"]
# get sample data from given function from [0, 20]
for _ in range(4):
    gap = int(index_gap[_])
    samples = np.linspace(0, 20, gap)
    index_sam[_] = func_given(samples)
    index_gap[_] = samples
    
result_array = np.zeros((4, 1000))
for _ in range(4):
    result_array[_] = np.array([cubic_spline(i, index_gap[_], index_sam[_]) for i in gap_test])
    
tlist = ["-", "--", ":", "-."]
plt.figure(figsize=(10, 10))
plt.grid()
plt.plot(gap_test, func_given(gap_test), color="black", alpha=0.5, linewidth=2)
for _ in range (4):
    plt.plot(gap_test, result_array[_], linestyle=tlist[_], alpha=0.7, linewidth=5)
plt.legend(["original function", "100 samples", "20 samples", "10 samples", "4 samples"])
plt.show()


