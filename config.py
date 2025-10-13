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

fre = 24e9 # 微波频率

# 电场网格参数
nx_e = 201   # 电场x方向网格数
ny_e = 201   # 电场y方向网格数
nz_e = 801   # 电场z方向网格数
nn_e = np.array([nx_e,ny_e,nz_e])

# 仿真区域边界
box_min = np.array([-0.0625, -0.0625, 0.0])    # 仿真区域最小坐标 [x_min, y_min, z_min]
box_max = np.array([0.0625, 0.0625, 0.422])    # 仿真区域最大坐标 [x_max, y_max, z_max]

# 数据文件路径
data_file = "data.h5"

# 场数据集名称
magnetic_field_dataset = "magnetic_fields"
electric_field_real_dataset = "electric_fields_real"
electric_field_imag_dataset = "electric_fields_imag"
ionization_dataset = "ionization"
excitation_dataset = "excitation"

# 粒子数量
npar_e = 5000  # 电子粒子数
weight_e = 1e7 # 电子宏粒子

# 时间参数
dt = 5e-12   # 时间步长 (秒)
Nt = 10000

# 输出和可视化参数
output_interval = 1000         # 输出间隔（时间步）
visualization_interval = 1000  # 可视化间隔（时间步）
save_final_state = True      # 是否保存最终状态
save_diagnostics = True      # 是否保存诊断数据
save_vtk = True             # 是否保存VTK格式
enable_realtime_vis = False  # 是否启用实时可视化（MPI环境建议关闭）

# 输出目录
output_dir = "output"
vtk_output_dir = "vtk_output"
diagnostics_dir = "diagnostics"

