import numpy as np
import math
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib import axes

def generate_F(scale_x,scale_y):
    F = np.zeros(shape=(x_len,y_len))
    with open('D:/学校/科研/ibmpgGen-1.spice', 'r') as file:
        for line in file.readlines():
            sp1 = line.split()
            if sp1[0][0] == 'i':#i
                sp2 = sp1[1].split('n')[1].split('_')
                x = float(sp2[1])/scale_x-1
                y = float(sp2[2])/scale_y-1
                F[int(x)][int(y)] = float(sp1[3].split('m')[0])
    return F
def generate_Z(n):
    Z = np.zeros(shape=(n,n))
    for i in range(n):
        for j in range(n):
            Z[i][j] = ((2/(n+1))**0.5)*math.sin(i*j*math.pi/(n+1))
    return Z
def generate_delta(n):
    delta = np.zeros(shape=(n,n))
    for i in range(n):
        for j in range(n):
            delta[i][j] = 2*(1-math.cos(i*math.pi/(n+1)))
    if delta[i][j] == np.nan:
        print('delta nan:', i, j)
    if delta[i][j] == np.inf:
        print('delta inf:', i, j)
    return delta
def generate_W(n,delta1,delta2,r1,r2,R):
    W = np.zeros(shape=(n,n))
    for i in range(n):
        for j in range(n):
            W[i][j] = (delta1[j][j]/r1 + delta2[i][i]/r2 + R*delta1[j][j]*delta2[i][i]/(r1*r2))**(-1)
    W[0][0] = 0

    return W
def generate_D2(z1,z2,F,W):
    d1 = W*np.dot(np.dot(z2,F),z1)
    d2 = np.dot(z2,d1)
    D = np.dot(d2,z1)
    return D
def show(D):
    plt.imshow(D)
    plt.tight_layout()
    plt.show()
def generate_pads(scale_x,scale_y):
    pads = np.zeros(shape=(x_len, y_len))
    with open('D:/学校/科研/ibmpgGen-1.spice', 'r') as file:
        for line in file.readlines():
            sp1 = line.split()
            if sp1[0][0] == 'v':
                sp2 = sp1[1].split('n')[1].split('_')
                x = float(sp2[1]) / scale_x - 1
                y = float(sp2[2]) / scale_y - 1
                pads[int(x)][int(y)] = 1
    return pads

x_len = 16
y_len = 16
R = 0
scale_x = 0.0001
scale_y = 0.0001
r_x = 0.000147089966
r_y = 0.000136162201

F = generate_F(scale_x,scale_y)
z1 = generate_Z(x_len)
z2 = generate_Z(y_len)
d1 = generate_delta(x_len)
d2 = generate_delta(y_len)
W = generate_W(x_len,d1,d2,r_x,r_y,R)
D2 =generate_D2(z1,z2,F,W)
pads = generate_pads(scale_x,scale_y)
print(pads)
show(D2)


