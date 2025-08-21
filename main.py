import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Add this import
import scipy



# def gather()


mass = 9.1 * 10e-31 # unit: kg
# dt = 1e-10 # unit: s
f = 28e9 #28 GHz
w = 2 * np.pi * f # unit: rad/s
charge = 1.6 * 10e-19 # unit: C

def boris_pusher(vel, electric_field, magnetic_field, q_m, dt):
    # Boris algorithm for velocity update
    v_minus = vel + electric_field * q_m * dt / 2
    t = magnetic_field * q_m * dt / 2
    t_mag = np.linalg.norm(t)
    # Cross product for rotation
    v_prime = v_minus + np.cross(v_minus, t)
    s = 2 * t / (1 + np.dot(t, t))
    v_plus = v_minus + np.cross(v_prime, s)
    vel_new = v_plus + electric_field * q_m * dt / 2
    return vel_new

times = 0 
Nt = 10

Np_electrons = 1
Np_argon = 1
Np_argon_1 = 1

# 定义粒子
electrons_vel = np.zeros((Np_electrons, 3))
electrons_pos = np.zeros((Np_electrons, 3))
electrons_mass = 1 # unit: electron_mass
electrons_charge = -1

argons_vel = np.zeros((Np_argon, 3))
argons_pos = np.zeros((Np_argon, 3))
argons_mass = 40 * 1836 # unit: proton_mass
argons_charge = 0

argons_1_vel = np.zeros((Np_argon_1, 3))
argons_1_pos = np.zeros((Np_argon_1, 3))
argons_1_mass = argons_mass
argons_1_charge = 1



# 定义几何
box_min = np.array([0, 0, 0]) # unit: m
box_max = np.array([10, 10, 10]) # unit: m

nx = 3
ny = 3
nz = 3

def get_local_pos(global_pos):
    local_pos = ((global_pos - box_min) / (box_max - box_min)) * np.array([nx, ny, nz])
    local_pos = np.clip(local_pos, 0, np.array([nx-1, ny-1, nz-1]))  # Ensure indices are within bounds
    return np.floor(local_pos).astype(int)

def gather(field, loc_pos):
    x, y, z = loc_pos
    return field[x, y, z]

def scatter(field, loc_pos, value):
    x, y, z = loc_pos
    field[x, y, z] += value
    return field

# 定义电场
ex_r = np.zeros((nx,ny,nz))
ey_r = np.zeros((nx,ny,nz))
ez_r = np.zeros((nx,ny,nz))


ex_i = np.zeros((nx,ny,nz))
ey_i = np.zeros((nx,ny,nz))
ez_i = np.zeros((nx,ny,nz))

e = np.zeros((nx,ny,nz,3))

e[:,:,:,2] = 0.0 # V/m

#定义磁场
bx = np.zeros((nx,ny,nz))
by = np.zeros((nx,ny,nz))
bz = np.zeros((nx,ny,nz))

b = np.zeros((nx,ny,nz,3))

## 初始化磁场
b[:,:,:,2] = 4 # Tesla

# 初始化粒子
## 初始位置
# electrons_pos = np.random.rand(Np_electrons, 3) * (box_max - box_min) + box_min
electrons_pos = np.array([[0.0, 0.0, 0.0]])
argons_1_pos = np.random.rand(Np_argon_1, 3) * (box_max - box_min) + box_min

## 初始速度
electrons_vel[0][0] = 100
argons_vel[:, 0]= 1000
argons_1_vel[:, 0] = 1000


# 推动粒子
## Boris
ex = np.zeros((nx,ny,nz))
ey = np.zeros((nx,ny,nz))
ez = np.zeros((nx,ny,nz))

# Store positions at each time step
positions_history = []
vel_history = []


b_mag = 4 # Tesla
period = 1 / (charge * b_mag/ mass/ (2 * np.pi))
print("cycltron frequency is ", 1/period, " Hz")
print("period is ", period, " s")

Nt = 166
# dt = period / Nt
dt = 5e-12
# Nt = int(period / dt) + 1
print("dt is ", dt, " s")


for i in range (Nt):
    for ip in range(Np_electrons):
        loc_pos = get_local_pos(electrons_pos[ip])
        e_loc = gather(e, loc_pos)
        b_loc = gather(b, loc_pos)
        electrons_vel[ip] = boris_pusher(electrons_vel[ip], e_loc, b_loc, charge/mass, dt)
        electrons_pos += electrons_vel * dt
        positions_history.append(electrons_pos.copy())
        print(f"time {i}, vel_x is {electrons_vel[ip][0]}, vel_y is {electrons_vel[ip][1]}, vel_z is {electrons_vel[ip][2]}")



# 统计粒子



# 求解possion方程



# 计算新场


# visualization
positions_history = np.array(positions_history)  # Shape: (Nt, Np_electrons, 3)

# 3D Scatter plot trajectory of the first electron
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(
    positions_history[:, 0, 0],  # x
    positions_history[:, 0, 1],  # y
    positions_history[:, 0, 2],  # z
    label='Electron 0',
    s=20
)
ax.set_xlabel('x position (m)')
ax.set_ylabel('y position (m)')
ax.set_zlabel('z position (m)')
ax.set_title('3D Scatter Trajectory of Electron 0')
ax.legend()
plt.show()