"""
PIC-MCC 仿真配置文件
包含网格参数、时间步长、边界条件等配置
"""

import numpy as np

# 网格参数
nx = 64      # x方向网格数
ny = 64      # y方向网格数  
nz = 65      # z方向网格数
nn = np.array([nx,ny,nz])

# 磁场网格参数
nx_b = 201   # 磁场x方向网格数
ny_b = 201   # 磁场y方向网格数
nz_b = 423   # 磁场z方向网格数
nn_b = np.array([nx_b,ny_b,nz_b])

# 电场网格参数
nx_e = 401   # 电场x方向网格数
ny_e = 401   # 电场y方向网格数
nz_e = 801   # 电场z方向网格数
nn_e = np.array([nx_e,ny_e,nz_e])

# 仿真区域边界
box_min = np.array([-0.0625, -0.0625, 0.0])    # 仿真区域最小坐标 [x_min, y_min, z_min]
box_max = np.array([0.0625, 0.0625, 0.422])    # 仿真区域最大坐标 [x_max, y_max, z_max]

# 时间参数
dt = 1e-11   # 时间步长 (秒)
Nt = 5000    # 总时间步数

# 数据文件路径
data_file = "data.h5"

# 场数据集名称
magnetic_field_dataset = "magnetic_fields"
electric_field_dataset = "electric_fields"
ionization_dataset = "ionization"
excitation_dataset = "excitation"


