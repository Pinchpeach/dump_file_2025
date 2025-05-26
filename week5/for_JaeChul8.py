import numpy as np
import matplotlib.pyplot as plt

# Read out data from "sn.dat" to array "SN_arr"
sn_dat = open("sn.dat", 'r')

SN = sn_dat.readlines()

sn_dat.close()

SN_arr = np.zeros((2, 16))

i = 0

for line in SN:
    string = line
    _, JD, Mag, dMag = line.split()
    if JD == "JD":
        continue
    SN_arr[0][i] = JD; SN_arr[1][i] = Mag
    i += 1
    
    cnt = (SN_arr[0][15] - SN_arr[0][0]) / 101
    
# define Lagrange_interpolation
def Lagrange_interpolation(data, x):
    y = 0
    
    for i in range(len(data[0])):
        up = 1; down = 1
        for j in range(len(data[0])):
            if i == j:
                continue
            up *= (x - data[0][j])
            down *= (data[0][i] - data[0][j])
        y += (data[1][i] * up / down)
    
    return y

# get result by Lagrange_interpolation
result_lag = np.zeros((2, 101))

x0 = SN_arr[0][0]

for _ in range(0, 101):
    result_lag[0][_] = x0
    result_lag[1][_] = Lagrange_interpolation(SN_arr, x0)
    x0 += cnt
    
# Plotting results by Lagrange interpolation
plt.figure()
plt.ylim(15, 20)
plt.gca().invert_yaxis()
plt.scatter(result_lag[0], result_lag[1], marker='o', alpha=0.7)
plt.scatter(SN_arr[0], SN_arr[1], marker='x')
plt.legend()

# noise in Lagrange interpolation
dmag = np.random.normal(0, 0.1, size=16)

result_lag_noise = np.zeros((2, 101))

SN_interrupt = SN_arr.copy()
SN_interrupt[1] += dmag

x0 = SN_arr[0][0]

for _ in range(0, 101):
    result_lag_noise[0][_] = x0
    result_lag_noise[1][_] = Lagrange_interpolation(SN_interrupt, x0)
    x0 += cnt
    
plt.figure()
plt.ylim(15, 20)
plt.gca().invert_yaxis()
plt.scatter(result_lag_noise[0], result_lag_noise[1], marker='o', alpha=0.7)
plt.scatter(SN_arr[0], SN_arr[1], marker='x')
plt.legend()

# difference between noise & without noise
RMSE_lag = np.sqrt(np.mean((result_lag[1] - result_lag_noise[1])**2))
print(RMSE_lag)

# define Neville's algorithm
def Neville_algorithm(data, x, start, end):
    if (end - start) == 0:
        return data[1][start]
    
    return (((x - data[0][end])*(Neville_algorithm(data, x, start, end - 1)) 
            + ((data[0][start] - x) * Neville_algorithm(data, x, start + 1, end))) 
            / (data[0][start] - data[0][end]))
    
result_nev = np.zeros((4, 2, 101))

adder = [2, 4, 6, 15]

for n in range(4):
    cur = 0
    x0 = SN_arr[0][0]
    for _ in range(len(result_nev[n][0])):
        if ((cur + adder[n] < 15) and (SN_arr[0][cur + 1] < x0)):
            cur += 1
        result_nev[n][0][_] = x0
        result_nev[n][1][_] = Neville_algorithm(SN_arr, x0, cur, cur+adder[n])
        x0 += cnt
        
# Plotting results by Lagrange interpolation
label = ['3 point', '5 point', '7 point', '16 point']
plt.figure(figsize=(10, 10))
plt.ylim(15, 20)
plt.xlim(2.454*10**6, 2.45408*10**6)
plt.grid()
plt.gca().invert_yaxis()
for _ in range(4):
    plt.scatter(result_nev[_][0], result_nev[_][1], marker='o', alpha=0.7, label=label[_])
plt.scatter(SN_arr[0], SN_arr[1], marker='x', color='red')
plt.legend()

#Gaussian noise 가내수공업
noise = np.random.normal(0, 0.1, size=16)
print(noise)

# Check out when data has noise.
# use Gaussian noise from above. set as "noise" with 1D, 16 float array
SN_arr_noise = np.zeros((2, 16))
SN_arr_noise += SN_arr
SN_arr_noise[1] += noise

#define array type data to save results of calculation
result_nev_noise = np.zeros((4, 2, 101))

adder_noise = [2, 4, 6, 15]

for k in range(4):
    x0 = SN_arr_noise[0][0]
    cur = 0
    for _ in range(len(result_nev_noise[k][0])):
        if ((cur + adder_noise[k] < 15) and (SN_arr_noise[0][cur + 1] < x0)):
            cur += 1
        result_nev_noise[k][0][_] = x0
        result_nev_noise[k][1][_] = Neville_algorithm(SN_arr_noise, x0, cur, cur+adder_noise[k])
        x0 += cnt    
        
plt.figure(figsize=(10, 10))
plt.grid(visible=True)
plt.ylim(15, 20)
plt.xlim(2.454*10**6, 2.45408*10**6)
plt.gca().invert_yaxis()
for _ in range(4):
    plt.plot(result_nev_noise[_][0], result_nev_noise[_][1], label=_)
plt.scatter(SN_arr_noise[0], SN_arr_noise[1], marker='x', color='red')
plt.scatter(SN_arr[0], SN_arr[1], marker='*', color='black')
plt.legend()

# 노이즈 유무에 따른 차이
RMSE = np.zeros(4)

for k in range(4):
    RMSE[k] = np.sqrt(np.mean(np.square(result_nev_noise[k][1] - result_nev[k][1])))
    
print(RMSE)