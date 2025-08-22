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

Np_electrons = 2
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
box_min = np.array([0., 0., 0.]) # unit: m
box_max = np.array([1., 1., 1.]) # unit: m

nx = 51
ny = 51
nz = 51

ne_grid = np.zeros([nx, ny, nz])  # Number of grid points in each dimension
    

def get_local_pos(global_pos):
    local_pos = ((global_pos - box_min) / (box_max - box_min)) * np.array([nx-1., ny-1., nz-1.])
    local_pos = np.clip(local_pos, 0.0, np.array([nx-1, ny-1, nz-1]))
    return local_pos

local_value_3d = 0
local_value_4d = np.zeros((3))

def gather(field, loc_pos):
    ix, iy, iz = loc_pos.astype(int)
    dx, dy, dz = loc_pos - np.array([ix, iy, iz])
    # 使用线性插值来获取局部场值
    if field.ndim == 3:
        local_value_3d = 0
        local_value_3d = field[ix, iy, iz] * (1 - dx) * (1 - dy) * (1 - dz) + \
                    field[ix + 1, iy, iz] * dx * (1 - dy) * (1 - dz) + \
                    field[ix, iy + 1, iz] * (1 - dx) * dy * (1 - dz) + \
                    field[ix, iy, iz + 1] * (1 - dx) * (1 - dy) * dz + \
                    field[ix + 1, iy + 1, iz] * dx * dy * (1 - dz) + \
                    field[ix, iy + 1, iz + 1] * (1 - dx) * dy * dz + \
                    field[ix + 1, iy, iz + 1] * dx * (1 - dy) * dz + \
                    field[ix + 1, iy + 1, iz + 1] * dx * dy * dz
        return local_value_3d
    
    elif field.ndim == 4:
        local_value_4d = np.zeros(3)
        local_value_4d[0] = gather(field[:, :, :, 0], loc_pos)
        local_value_4d[1] = gather(field[:, :, :, 1], loc_pos)
        local_value_4d[2] = gather(field[:, :, :, 2], loc_pos)
        return local_value_4d
    else:
        raise ValueError("Field must be 3D or 4D array.")

def scatter(field, loc_pos, value):
    ix, iy, iz = loc_pos.astype(int)
    dx, dy, dz = loc_pos - np.array([ix, iy, iz])
    # 使用线性插值来更新局部场值
    if field.ndim == 3:
        field[ix, iy, iz] += value * (1 - dx) * (1 - dy) * (1 - dz)
        field[ix + 1, iy, iz] += value * dx * (1 - dy) * (1 - dz)
        field[ix, iy + 1, iz] += value * (1 - dx) * dy * (1 - dz)
        field[ix, iy, iz + 1] += value * (1 - dx) * (1 - dy) * dz
        field[ix + 1, iy + 1, iz] += value * dx * dy * (1 - dz)
        field[ix, iy + 1, iz + 1] += value * (1 - dx) * dy * dz
        field[ix + 1, iy, iz + 1] += value * dx * (1 - dy) * dz
        field[ix + 1, iy + 1, iz + 1] += value * dx * dy * dz
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
electrons_pos = np.random.rand(Np_electrons, 3) * (box_max - box_min) + box_min
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
        electrons_pos[ip] += electrons_vel[ip] * dt
        # electrons_pos = boundary_check(electrons_pos)
        positions_history.append(electrons_pos.copy())
        times +=dt



# 统计粒子
for ip in range(Np_electrons):
    loc_pos = get_local_pos(electrons_pos[ip])
    ne_grid = scatter(ne_grid, loc_pos, electrons_charge)




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