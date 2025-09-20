import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Add this import
import h5py
import scipy


def main():
    # module setting
    np.random.seed(0)
    
    # physics parameters
    mass = 9.1 * 10e-31 # unit: kg
    # dt = 1e-10 # unit: s
    f = 28e9 #28 GHz
    w = 2 * np.pi * f # unit: rad/s
    charge = 1.6 * 10e-19 # unit: C


    dt = 5e-12
    print("dt is ", dt, " s")

    times = 0 # real time
    Nt = 1000 # time step

    # puhser
    def boris_pusher(vel, electric_field, magnetic_field, q_m, dt):
        # Boris algorithm for velocity update
        v_minus = vel + electric_field * q_m * dt / 2
        t = magnetic_field * q_m * dt / 2
        # Cross product for rotation
        v_prime = v_minus + np.cross(v_minus, t)
        s = 2 * t / (1 + np.dot(t, t))
        v_plus = v_minus + np.cross(v_prime, s)
        vel_new = v_plus + electric_field * q_m * dt / 2
        return vel_new 

    # 定义粒子
    N_species = 3
    Np_electrons = 100
    Np_argons = 1
    Np_argons_1 = 1

    electrons_vel = np.zeros((Np_electrons, 3))
    electrons_pos = np.zeros((Np_electrons, 3))
    electrons_mass = 1 # unit: electron_mass
    electrons_charge = -1

    argons_vel = np.zeros((Np_argons, 3))
    argons_pos = np.zeros((Np_argons, 3))
    argons_mass = 40 * 1836 # unit: proton_mass
    argons_charge = 0

    argons_1_vel = np.zeros((Np_argons_1, 3))
    argons_1_pos = np.zeros((Np_argons_1, 3))
    argons_1_mass = argons_mass
    argons_1_charge = 1



    # 定义几何
    box_min = np.array([-0.0625, -0.0625, 0.]) # unit: m
    box_max = np.array([0.0625, 0.0625, 0.422]) # unit: m

    nx = 64
    ny = 64
    nz = 65
    
    gird_x, gird_y, gird_z = np.mgrid[box_min[0]:box_max[0]:nx*1j, box_min[1]:box_max[1]:ny*1j, box_min[2]:box_max[2]:nz*1j]

    ne_grid = np.zeros([nx, ny, nz])  # Number of grid points in each dimension
        

    def get_local_pos(global_pos, nx, ny, nz):
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

    # boundary check
    def boundary_check(pos, vel):
        """
        检查粒子边界条件并处理边界碰撞
        
        参数:
        pos: 粒子位置数组 (N, 3)
        vel: 粒子速度数组 (N, 3)
        
        返回:
        pos: 修正后的位置
        vel: 修正后的速度
        """
        # 检查每个维度的边界
        for dim in range(3):
            # 检查下边界
            mask_lower = pos[:, dim] < box_min[dim]
            pos[mask_lower, dim] = box_min[dim]
            vel[mask_lower, dim] = -vel[mask_lower, dim]  # 反射速度
            
            # 检查上边界
            mask_upper = pos[:, dim] > box_max[dim]
            pos[mask_upper, dim] = box_max[dim]
            vel[mask_upper, dim] = -vel[mask_upper, dim]  # 反射速度
        
        return pos, vel
        # 如果超出边界，则将其放置到边界上，如果没有超出边界，则不做修改。返回的仍然是一个Pos
            

    # 定义电场
    nx_e = 201
    ny_e = 201
    nz_e = 801
    ex_r = np.zeros((nx_e,ny_e,nz_e))
    ey_r = np.zeros((nx_e,ny_e,nz_e))
    ez_r = np.zeros((nx_e,ny_e,nz_e))


    ex_i = np.zeros((nx_e,ny_e,nz_e))
    ey_i = np.zeros((nx_e,ny_e,nz_e))
    ez_i = np.zeros((nx_e,ny_e,nz_e))

    e = np.zeros((nx_e,ny_e,nz_e,3))

    e[:,:,:,2] = 0.0 # V/ms

    #定义磁场
    nx_b = 201
    ny_b = 201
    nz_b = 423
    b = np.zeros((nx_b,ny_b,nz_b,3))


    ## 初始化磁场
    ### TODO：利用h5py读取磁场  
    with h5py.File("data.h5", "r") as f:
        b = f["magnetic_fields"][...]
    
    print("magnetic field loaded")
       
    # 初始化粒子
    ## 初始位置
    electrons_pos = np.random.rand(Np_electrons, 3) * (box_max - box_min) + box_min
    argons_1_pos = np.random.rand(Np_argons_1, 3) * (box_max - box_min) + box_min

    ## 初始速度
    electrons_vel[:, 0] = 100
    argons_vel[:, 0]= 1000
    argons_1_vel[:, 0] = 1000


    # 推动粒子
    ## Boris
    ex = np.zeros((nx,ny,nz))
    ey = np.zeros((nx,ny,nz))
    ez = np.zeros((nx,ny,nz))

    # Store positions at each time step
    electrons_positions_history = []
    argons_1_positions_history = []
    vel_history = []



    # period = 1 / (charge * b_mag/ mass/ (2 * np.pi))
    # print("cycltron frequency is ", 1/period/1e9, " GHz")
    # print("period is ", period, " s")


    # pusher
    ## push electrons
    for i in range (Nt):
        for ip in range(Np_electrons):
            loc_pos = get_local_pos(electrons_pos[ip], nx ,ny, nz)
            loc_pos_e = get_local_pos(electrons_pos[ip], nx_e ,ny_e, nz_e)
            e_loc = gather(e, loc_pos)
            loc_pos_b = get_local_pos(electrons_pos[ip], nx_b ,ny_b, nz_b)
            b_loc = gather(b, loc_pos_b)
            electrons_vel[ip] = boris_pusher(electrons_vel[ip], e_loc, b_loc, charge/mass, dt)
            electrons_pos[ip] += electrons_vel[ip] * dt
            times +=dt
        # 边界检查
        electrons_pos, electrons_vel = boundary_check(electrons_pos, electrons_vel)
        electrons_positions_history.append(electrons_pos.copy())


    ## push argon 1+
    for i in range (Nt):
        for ip in range(Np_argons_1):
            loc_pos = get_local_pos(argons_1_pos[ip], nx ,ny, nz)
            loc_pos_e = get_local_pos(argons_1_pos[ip], nx_e ,ny_e, nz_e)
            e_loc = gather(e, loc_pos)
            loc_pos_b = get_local_pos(argons_1_pos[ip], nx_b ,ny_b, nz_b)
            b_loc = gather(b, loc_pos)
            argons_1_vel[ip] = boris_pusher(argons_1_vel[ip], e_loc, b_loc, -charge/(40 * mass * 1836), dt)
            argons_1_pos[ip] += argons_1_vel[ip] * dt
            times +=dt
        # 边界检查
        argons_1_pos, argons_1_vel = boundary_check(argons_1_pos, argons_1_vel)
        argons_1_positions_history.append(argons_1_pos.copy())



    # 统计粒子
    for ip in range(Np_electrons):
        loc_pos = get_local_pos(electrons_pos[ip], nx, ny, nz)
        ne_grid = scatter(ne_grid, loc_pos, electrons_charge)

    for ip in range(Np_argons_1):
        loc_pos = get_local_pos(argons_1_pos[ip], nx, ny, nz)
        ne_grid = scatter(ne_grid, loc_pos, argons_1_charge)


    charge_total = 0.
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                if ne_grid[ix,iy,iz] != 0:
                    charge_total += ne_grid[ix,iy,iz]
                    print(f"ne_grid {ix},{iy},{iz} is {ne_grid[ix,iy,iz]}")

    print(f"total charge is {charge_total}")



    # 求解possion方程



    # 计算新场


    # visualization
    electrons_positions_history = np.array(electrons_positions_history)  # Shape: (Nt, Np_electrons, 3)
    argons_1_positions_history = np.array(argons_1_positions_history)  # Shape: (Nt, Np_argons_1, 3)

    # 3D Scatter plot trajectory of the first electron
    fig = plt.figure(figsize=(8,6))
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(
        electrons_positions_history[:, 0, 0] * 1000,  # x
        electrons_positions_history[:, 0, 1] * 1000,  # y
        electrons_positions_history[:, 0, 2] * 1000,  # z
        label='Electron 0',
        s=20
    )
    ax1.set_xlabel('x(mm)')
    ax1.set_ylabel('y(mm)')
    ax1.set_zlabel('z(mm)')
    ax1.set_title('3D Scatter Trajectory of Electron 0')
    ax1.legend()

    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(
        argons_1_positions_history[:, 0, 0] * 1000,  # x
        argons_1_positions_history[:, 0, 1] * 1000,  # y
        argons_1_positions_history[:, 0, 2] * 1000,  # z
        label='argons_1 0',
        s=20
    )
    ax2.set_xlabel('x(mm)')
    ax2.set_ylabel('y(mm)')
    ax2.set_zlabel('z(mm)')
    ax2.set_title('3D Scatter Trajectory of Argon_1 0')
    ax2.legend()
    plt.show()
    
    f.close()


if __name__ == "__main__":
    main()