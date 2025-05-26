import numpy as np
import matplotlib.pyplot as plt

#read file
sn_dat = open("sn.dat", 'r')

SN = sn_dat.readlines()

sn_dat.close()

SN_arr = np.zeros((2, 16))
Noise = np.zeros((1, 16))

i = 0

for line in SN:
    string = line
    _, JD, Mag, dMag = line.split()
    if JD == "JD":
        continue
    SN_arr[0][i] = JD; SN_arr[1][i] = Mag; Noise[0][i] = dMag
    i += 1
    
#define linear interpolation
def linear_interpolation(pos, x1, x2, y1, y2):
    w1 = (x2 - pos) / (x2 - x1)
    w2 = (pos - x1) / (x2 - x1)
    
    return w1 * y1 + w2 * y2

#define n to make 100 points between 최소JD and 최대JD
n = (SN_arr[0][15] - SN_arr[0][0]) / 101

result = np.zeros((2, 100))
cnt = 0
x = SN_arr[0][0]
x += n

#do linear interpolation based on given dataset
for _ in range(100):
    result[0][_] = x
    
    if (x > SN_arr[0][cnt + 1]):
        cnt += 1
        if cnt == 15:
            x = SN_arr[0][15]
    
    x1 = SN_arr[0][cnt]; y1 = SN_arr[1][cnt]
    x2 = SN_arr[0][cnt+1]; y2 = SN_arr[1][cnt+1]
    result[1][_] = linear_interpolation(x, x1, x2, y1, y2)
    x += n

#plotting the result without noise
plt.figure(figsize=(12, 12))
plt.gca().invert_yaxis()
plt.scatter(result[0], result[1], marker='o', alpha=0.7)
plt.scatter(SN_arr[0], SN_arr[1], marker='x')
plt.legend()

# Check out when noise is added
# add noise to given data(dMag)
SN_arr_noise = np.zeros((2,16))
SN_arr_noise = SN_arr
SN_arr_noise[1] += Noise[0]

# with noise, do linear interpolation again
result_n = np.zeros((2, 100))
cnt = 0
x = SN_arr_noise[0][0]
x += n

for _ in range(100):
    result_n[0][_] = x
    
    if (x > SN_arr_noise[0][cnt + 1]):
        cnt += 1
        if cnt == 15:
            x = SN_arr_noise[0][15]
    
    x1 = SN_arr_noise[0][cnt]; y1 = SN_arr_noise[1][cnt]
    x2 = SN_arr_noise[0][cnt+1]; y2 = SN_arr_noise[1][cnt+1]
    result_n[1][_] = linear_interpolation(x, x1, x2, y1, y2)
    x += n

# plotting with the result
plt.figure(figsize=(12, 12))
plt.gca().invert_yaxis()
plt.scatter(result_n[0], result_n[1], marker='o', alpha=0.7)
plt.scatter(SN_arr_noise[0], SN_arr_noise[1], marker='x')
plt.legend()

# result 차이 비교(noise 유무에 따른 비교)
mse = np.mean(np.square(result[1] - result_n[1]))

print(mse)