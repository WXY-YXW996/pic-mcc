"""
PIC-MCC 集成仿真主程序
包含可视化、输出和诊断功能
"""

from mpi4py import MPI
import numpy as np
import h5py
import pandas as pd
from scipy.interpolate import griddata
import numba
import matplotlib.pyplot as plt
import time
import os

import World
import Field
import Species
import Source
import Parameters
import config  # 导入配置文件

# 导入新增的模块
import Visualization
import OutputModule
import Diagnostics

# 从配置文件导入参数
nn = config.nn
nn_b = config.nn_b
nn_e = config.nn_e
box_min, box_max = config.box_min, config.box_max
dt, Nt = config.dt, config.Nt
npar_e = config.npar_e

# 导入输出和可视化参数
output_interval = config.output_interval
visualization_interval = config.visualization_interval
save_final_state = config.save_final_state
save_diagnostics = config.save_diagnostics
save_vtk = config.save_vtk
enable_realtime_vis = config.enable_realtime_vis

##########################

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

start_time = time.time()

# 初始化仿真世界
world = World.World(nn)
world.setBox(box_min, box_max)
world.setTime(dt, Nt)

# 初始化磁场
b = Field.Field(nn_b)
b.setBox(world.box_min, world.box_max)
b.load(config.data_file, config.magnetic_field_dataset)

# 初始化电场
e = Field.Field(nn_e)
e.setBox(world.box_min, world.box_max)
# e.load(config.data_file, config.electric_field_dataset)

# 初始化物种
species = [Species.Species("e-", Parameters.ME, -1. * Parameters.QE, b, e, world)]

# 初始化源
source = [Source.Source(sp, world) for sp in species]
source[0].sample(npar_e)

# 初始化可视化和输出模块
if rank == 0:
    print("初始化可视化和输出模块...")

# 静态可视化器
visualizer = Visualization.Visualizer(world, config.output_dir)

# VTK输出器
if save_vtk:
    vtk_output = OutputModule.VTKOutput(config.vtk_output_dir, "pic_mcc")

# 诊断器
diagnostics = Diagnostics.PhysicalDiagnostics(world)

# 诊断输出器
if save_diagnostics:
    diag_output = OutputModule.DiagnosticOutput(config.diagnostics_dir, "pic_mcc")

# 实时可视化（可选）
if enable_realtime_vis:
    realtime_vis = Visualization.RealTimeVisualizer(world, visualization_interval)
    realtime_diag = Diagnostics.RealTimeDiagnostics(world, visualization_interval)

if rank == 0:
    print(f"仿真开始，共 {Nt} 个时间步")
    print(f"输出间隔: {output_interval} 步")
    print(f"可视化间隔: {visualization_interval} 步")
    print(f"粒子数: {sum([sp.npar for sp in species])}")

# 主仿真循环
while world.advanceTime():
    # 推进粒子
    for sp in species:
        sp.advance()
        sp.boundary()
    
    # 计算密度
    species[0].computeNumberDensity()
    
    # 收集诊断数据
    fields_dict = {
        'magnetic': b,
        'electric': e,
        'density': species[0].den
    }
    
    # 更新诊断历史
    diagnostics.update_history(species, fields_dict)
    
    # 定期保存诊断数据
    if save_diagnostics and world.ts % output_interval == 0:
        diag_output.collect_timestep_data(world, species, fields_dict)
    
    # 定期输出VTK文件
    if save_vtk and world.ts % output_interval == 0:
        # 保存粒子数据
        vtk_output.write_unstructured_grid(species, world.ts)
        
        # 保存场数据
        vtk_output.write_structured_grid(b, world.ts, "magnetic_field", "vector")
        vtk_output.write_structured_grid(species[0].den, world.ts, "density", "scalar")
        
        # 创建并行集合文件（只有根进程）
        vtk_output.write_parallel_collection(world.ts, "particles")
        vtk_output.write_parallel_collection(world.ts, "fields")
    
    # 定期可视化
    if world.ts % visualization_interval == 0:
        # 保存静态图像
        visualizer.save_timestep_plots(species, fields_dict)
        
        # 实时可视化更新
        if enable_realtime_vis:
            realtime_vis.update(species, fields_dict)
            realtime_diag.update(diagnostics, species)
    
    # 进度输出
    if rank == 0 and world.ts % (Nt // 10) == 0:
        elapsed_time = time.time() - start_time
        progress = world.ts / Nt * 100
        estimated_total = elapsed_time / (world.ts / Nt)
        remaining_time = estimated_total - elapsed_time
        
        print(f"进度: {progress:.1f}% (时间步 {world.ts}/{Nt})")
        print(f"已用时间: {elapsed_time:.1f}s, 预计剩余: {remaining_time:.1f}s")
        
        # 输出当前物理量
        kinetic = diagnostics.compute_kinetic_energy(species)
        momentum = diagnostics.compute_momentum(species)
        temperatures = diagnostics.compute_temperature(species)
        
        print(f"总动能: {kinetic['total']:.2e} J")
        print(f"总动量: {momentum['magnitude']:.2e} kg⋅m/s")
        print(f"电子温度: {temperatures[0]:.2e} K")
        print("-" * 50)

# 仿真结束后的处理
if rank == 0:
    print("仿真完成，正在保存最终数据...")

# 保存最终状态
if save_final_state:
    if save_diagnostics:
        diag_output.save_final_state(world, species, fields_dict)
        diag_output.save_time_series()
    
    # 保存最终诊断报告
    final_diag_file = f"{config.diagnostics_dir}/final_diagnostics.h5"
    diagnostics.save_diagnostics(final_diag_file, species, fields_dict)

# 生成最终可视化
if rank == 0:
    print("生成最终可视化图表...")
    
    # 生成最终粒子分布图
    for plane in ['xy', 'xz', 'yz']:
        save_path = f"{config.output_dir}/final_particles_{plane}.png"
        visualizer.plot_particle_distribution_2d(species, plane=plane, 
                                                save_path=save_path, 
                                                show_velocity=True)
    
    # 生成最终场分布图
    for field_name, field_obj in fields_dict.items():
        if field_obj.field is not None:
            for plane in ['xy', 'xz', 'yz']:
                if len(field_obj.field.shape) == 4:  # 矢量场
                    for comp in range(3):
                        comp_names = ['x', 'y', 'z']
                        save_path = f"{config.output_dir}/final_{field_name}_{comp_names[comp]}_{plane}.png"
                        visualizer.plot_field_2d(field_obj, component=comp, plane=plane, 
                                               save_path=save_path, 
                                               title=f'Final {field_name.capitalize()} Field {comp_names[comp].upper()}')
                else:  # 标量场
                    save_path = f"{config.output_dir}/final_{field_name}_{plane}.png"
                    visualizer.plot_field_2d(field_obj, plane=plane, save_path=save_path, 
                                           title=f'Final {field_name.capitalize()} Field')
    
    # 生成能量演化图
    save_path = f"{config.output_dir}/final_energy_evolution.png"
    visualizer.plot_particle_energy_evolution(species, save_path=save_path)

# 创建ParaView状态文件
if save_vtk and rank == 0:
    timesteps = list(range(0, Nt+1, output_interval))
    OutputModule.create_paraview_state_file(config.vtk_output_dir, timesteps)

# 关闭实时可视化
if enable_realtime_vis:
    realtime_vis.close()
    realtime_diag.close()

# 输出仿真统计信息
if rank == 0:
    total_time = time.time() - start_time
    print(f"\n{'='*50}")
    print("仿真完成统计:")
    print(f"总运行时间: {total_time:.2f} 秒")
    print(f"平均每时间步: {total_time/Nt:.4f} 秒")
    print(f"最终时间: {world.time:.2e} 秒")
    print(f"最终时间步: {world.ts}")
    
    # 最终物理量
    final_kinetic = diagnostics.compute_kinetic_energy(species)
    final_momentum = diagnostics.compute_momentum(species)
    final_temperatures = diagnostics.compute_temperature(species)
    
    print(f"\n最终物理量:")
    print(f"总动能: {final_kinetic['total']:.2e} J")
    print(f"总动量: {final_momentum['magnitude']:.2e} kg⋅m/s")
    print(f"电子温度: {final_temperatures[0]:.2e} K")
    
    # 粒子统计
    print(f"\n粒子统计:")
    for i, sp in enumerate(species):
        global_npar = comm.allreduce(sp.npar, op=MPI.SUM)
        print(f"  {sp.name}: {global_npar} 个粒子")
    
    print(f"\n输出文件位置:")
    print(f"  图像输出: {config.output_dir}/")
    if save_vtk:
        print(f"  VTK文件: {config.vtk_output_dir}/")
    if save_diagnostics:
        print(f"  诊断数据: {config.diagnostics_dir}/")
    
    print(f"{'='*50}\n")

MPI.Finalize()

# 添加一个简单的使用说明
if rank == 0:
    print("使用说明:")
    print("1. 查看生成的PNG图像了解仿真结果")
    print("2. 使用ParaView打开VTK文件进行高级可视化")
    print("3. 分析HDF5诊断文件获取详细数据")
    print("4. 修改config.py中的参数调整输出行为")
