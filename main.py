import os
import numpy as np
import matplotlib as plt
import scipy

mass = 9.1 * 10e-31 # unit: kg
dt = 1e-4 # unit: s
f = 28e9 #28 GHz
w = 2 * np.pi * f # unit: rad/s
charge = 1.6 * 10e-19 # unit: C

times = 0 
Nt = 1000

# 定义粒子
electrons_vel = np.zeros((100, 3))
electrons_pos = np.zeros((100, 3))
electrons_mass = 1 # unit: electron_mass
electrons_charge = -1

argons_vel = np.zeros((100, 3))
argons_pos = np.zeros((100, 3))
argons_mass = 40 * 1836 # unit: proton_mass
argons_charge = 0

argons_1_vel = np.zeros((100, 3))
argons_1_pos = np.zeros((100, 3))
argons_1_mass = argons_mass
argons_1_charge = 1



# 定义几何
box_min = np.array([0, 0, 0]) # unit: m
box_max = np.array([10, 10, 10]) # unit: m

nx = 100
ny = 100
nz = 100

# 定义电场
ex_r = np.zeros((nx, 3))
ey_r = np.zeros((ny, 3))
ez_r = np.zeros((nz, 3))


ex_i = np.zeros((nx, 3))
ey_i = np.zeros((ny, 3))
ez_i = np.zeros((nz, 3))


#定义磁场
bx = np.zeros((nx, 3))
by = np.zeros((ny, 3))
bz = np.zeros((nz, 3))

bz[:, :] = 1

magnetic_field = (bx, by, bz)
## 初始化粒子
electrons_pos = np.random.rand(100, 3) * (box_max - box_min) + box_min
argons_1_pos = np.random.rand(100, 3) * (box_max - box_min) + box_min


electrons_vel[:0] = 1000
argons_vel[:0] = 1000
argons_1_vel[:0] = 1000


# 推动粒子
## Boris
ex = np.zeros((nx,3))
ey = np.zeros((ny,3))
ez = np.zeros((nz,3))
electric_field = (ex, ey, ez)

v_minus = vel + charge / mass * electric_field * dt / 2
t = charge / mass  * magnetic_field * dt / 2
v_p = 





# 统计粒子



# 求解possion方程



# 计算新场