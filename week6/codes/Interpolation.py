import numpy as np
import matplotlib.pyplot as plt

#define linear interpolation
def linear_interpolation(pos, x1, x2, y1, y2):
    w1 = (x2 - pos) / (x2 - x1)
    w2 = (pos - x1) / (x2 - x1)
    
    return w1 * y1 + w2 * y2

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

def Neville_algorithm(data, x, start, end):
    if (end - start) == 0:
        return data[1][start]
    
    return (((x - data[0][end])*(Neville_algorithm(data, x, start, end - 1)) + ((data[0][start] - x) * Neville_algorithm(data, x, start + 1, end))) / (data[0][start] - data[0][end]))