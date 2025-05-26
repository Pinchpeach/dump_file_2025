import numpy as np
import matplotlib.pyplot as plt

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
    
JD_norm = SN_arr[0] - SN_arr[0][0]
Mag_dat = SN_arr[1]

derv_two = np.zeros(14).T

mat_cal = np.zeros((14, 14))

res_cal = np.zeros(14).T

for _ in range(len(JD_norm) - 2):
    mat_cal[_][_] = (JD_norm[_ + 2] - JD_norm[_]) / 3
    if _ != 0:
        mat_cal[_][_ - 1] = (JD_norm[_ + 1] - JD_norm[_]) / 6
    if _ != len(JD_norm) - 3:
        mat_cal[_][_ + 1] = (JD_norm[_ + 2] - JD_norm[_ + 1]) / 6
    
    res_cal[_] = ((Mag_dat[_ + 2] - Mag_dat[_ + 1]) / (JD_norm[_ + 2] - JD_norm[_ + 1])) 
    - ((Mag_dat[_ + 1] - Mag_dat[_]) / (JD_norm[_ + 1] - JD_norm[_]))

inv_mat_cal = np.linalg.inv(mat_cal)
derv_two = (res_cal) @ inv_mat_cal
derv_two = np.append(derv_two, 0)
derv_two = np.append(0, derv_two)

# algorithm to find position, return index of the low 
def find_pos(x, JD_norm):
    cpy_JD = JD_norm.copy()

    low = 0
    high = len(cpy_JD) - 1

    while high - low > 1:
        mid = (low + high) // 2
        if JD_norm[mid] > x:
            high = mid
        else:
            low = mid
    
    return low

# since we calculate 2nd dervation of our future graph, we can now interpolate using cubic spline interpolation

def cubic_spline(x, JD_norm, Mag_dat, derv_two):
    # initial setup

    index = find_pos(x, JD_norm)

    x0 = JD_norm[index]
    x1 = JD_norm[index + 1]
    
    y0 = Mag_dat[index]
    y1 = Mag_dat[index + 1]

    t = x1 - x0
    A = (x1 - x) / t
    B = (x - x0) / t
    C = (A**3 - A) * (t**2) / 6
    D = (B**3 - B) * (t**2) / 6

    return A * y0 + B * y1 + C * derv_two[index] + D * derv_two[index + 1]  

plt.figure(figsize=(10, 10))
plt.gca().invert_yaxis()
x_axis = np.linspace(JD_norm[0], JD_norm[15], 1000)
y_axis = np.array([cubic_spline(x, JD_norm, Mag_dat, derv_two) for x in x_axis])
plt.scatter(JD_norm, Mag_dat)
plt.plot(x_axis, y_axis)
plt.show()

def first_derv(x, JD_norm, Mag_dat, derv_two):
    index = find_pos(x, JD_norm)
    x0 = JD_norm[index]
    x1 = JD_norm[index + 1]
    y0 = Mag_dat[index]
    y1 = Mag_dat[index + 1]
    
    A = (x1 - x) / (x1 - x0)
    B = (x - x0) / (x1 - x0)
    
    d_x = x1 - x0
    d_y = y1 - y0
    
    return (d_y / d_x) - ((3 * A**2 - 1)/6)*derv_two[index]*d_x + ((3 * B**2 - 1)/6)*derv_two[index + 1]*d_x

def second_derv(x, JD_norm, Mag_dat, derv_two):
    index = find_pos(x, JD_norm)
    x0 = JD_norm[index]
    x1 = JD_norm[index + 1]
    y0 = Mag_dat[index]
    y1 = Mag_dat[index + 1]
    
    A = (x1 - x) / (x1 - x0)
    B = (x - x0) / (x1 - x0)
    
    return A * derv_two[index] + B * derv_two[index + 1]

y_fir = np.array([first_derv(x, JD_norm, Mag_dat, derv_two) for x in x_axis])
y_sec = np.array([second_derv(x, JD_norm, Mag_dat, derv_two) for x in x_axis])

plt.figure()
plt.gca().invert_yaxis()
plt.plot(x_axis, y_fir, label="First Derivative")
plt.plot(x_axis, y_sec, label="Second Derivative")
plt.legend()

import Interpolation as ip

result = np.zeros((2, 2, 1000))
cnt = 0
for _ in range(1000):    
    x = x_axis[_]
    if (x > JD_norm[cnt + 1]):
        cnt += 1
        if cnt == 15:
            x = JD_norm[15]
    
    x1 = JD_norm[cnt]; y1 = Mag_dat[cnt]
    x2 = JD_norm[cnt+1]; y2 = Mag_dat[cnt+1]
    result[0][1][_] = ip.linear_interpolation(x, x1, x2, y1, y2)
    result[0][0][_] = x
    
cnt = x_axis[1] - x_axis[0]
x0 = SN_arr[0][0]
for _ in range(1000):
    result[1][0][_] = x0
    result[1][1][_] = ip.Neville_algorithm(SN_arr, x0, 0, 15)
    x0 += cnt

result[1][0] -= SN_arr[0][0]

plt.figure(figsize=(20, 20))
plt.ylim(15, 20)
plt.gca().invert_yaxis()
plt.plot(result[0][0], result[0][1], label="Linear Interpolation")
plt.plot(result[1][0], result[1][1], label="Neville Interpolation")
plt.plot(x_axis, y_axis, label="Cubic Spline Interpolation")
plt.scatter(JD_norm, Mag_dat, label="Data")
plt.legend()


#############################################################################

import numpy as np
import matplotlib.pyplot as plt
import astropy.io.fits as fits
import scipy.interpolate as ip

pic = fits.open('image3.fits')
print(type(pic))
pic.info()
pic[0].header

data_arr = pic[0].data

plt.imshow(data_arr)

from scipy.interpolate import griddata

x = np.linspace(0, 1, data_arr.shape[1])
y = np.linspace(0, 1, data_arr.shape[0])
X, Y = np.meshgrid(x, y)

x_new = np.linspace(0, 1, 500)
y_new = np.linspace(0, 1, 500)
X_new, Y_new = np.meshgrid(x_new, y_new)

result_nearest = griddata((X.flatten(), Y.flatten()), data_arr.flatten(), (X_new, Y_new), method='nearest')
result_linear = griddata((X.flatten(), Y.flatten()), data_arr.flatten(), (X_new, Y_new), method='linear')
result_cubic = griddata((X.flatten(), Y.flatten()), data_arr.flatten(), (X_new, Y_new), method='cubic')

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.title("Nearest Neighbor")
plt.imshow(result_nearest)

plt.subplot(1, 3, 2)
plt.title("Bilinear")
plt.imshow(result_linear)

plt.subplot(1, 3, 3)
plt.title("Bicubic")
plt.imshow(result_cubic)

plt.show()