# -*- coding: utf-8 -*-
"""
泊松方程求解器使用示例

这个脚本展示了如何使用 World 类中的 potentialSlover 方法
来求解泊松方程：∇²φ = -ρ/ε₀

作者: PIC-MCC 仿真团队
"""

import numpy as np
import World
import config

config.nn = (10, 10, 10)
config.box_min = np.array([0.0, 0.0, 0.0])
config.box_max = np.array([1.0, 1.0, 1.0])

def create_test_charge_distribution():
    """
    创建一个测试电荷分布
    这里创建一个简单的高斯分布作为示例
    """
    # 使用配置文件中的参数
    world = World.World(config.nn, config.box_min, config.box_max)
    
    # 初始化电荷密度场
    nx, ny, nz = config.nn
    x_center = (config.box_max[0] + config.box_min[0]) / 2
    y_center = (config.box_max[1] + config.box_min[1]) / 2
    z_center = (config.box_max[2] + config.box_min[2]) / 2
    
    # 创建高斯电荷分布
    sigma = 0.1  # 高斯分布标准差 (米)
    charge_density = 1e6  # 电荷密度 (C/m³)
    
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                # 计算网格点的物理坐标
                x = config.box_min[0] + i * world.dh[0]
                y = config.box_min[1] + j * world.dh[1]
                z = config.box_min[2] + k * world.dh[2]
                
                # 计算到中心的距离
                r_squared = (x - x_center)**2 + (y - y_center)**2 + (z - z_center)**2
                
                # 高斯分布
                rho_value = charge_density * np.exp(-r_squared / (2 * sigma**2))
                world.rho.field[i, j, k, 0] = rho_value
    
    return world

def run_poisson_solver_example():
    """
    运行泊松方程求解器示例
    """
    print("=== 泊松方程求解器示例 ===")
    print("1. 创建测试电荷分布...")
    
    # 创建带有测试电荷分布的世界
    world = create_test_charge_distribution()
    
    print(f"   网格尺寸: {world.nn}")
    print(f"   仿真区域: {world.box_min} 到 {world.box_max}")
    print(f"   网格间距: {world.dh}")
    
    # 输出电荷密度统计信息
    rho_stats = world.rho.getFieldStats()
    print(f"   电荷密度统计:")
    print(f"     最大值: {rho_stats['max']:.2e} C/m³")
    print(f"     最小值: {rho_stats['min']:.2e} C/m³")
    print(f"     平均值: {rho_stats['mean']:.2e} C/m³")
    
    print("\n2. 求解泊松方程...")
    try:
        # 调用泊松方程求解器（使用内存优化选项）
        print("   使用迭代求解器以节省内存...")
        world.potentialSlover()
        
        print("   求解完成！")
        
        # 输出电势统计信息
        phi_stats = world.potential.getFieldStats()
        print(f"   电势统计:")
        print(f"     最大值: {phi_stats['max']:.2e} V")
        print(f"     最小值: {phi_stats['min']:.2e} V")
        print(f"     平均值: {phi_stats['mean']:.2e} V")
        
        # 输出电场统计信息
        if hasattr(world, 'electric_field'):
            E_stats = world.electric_field.getFieldStats()
            print(f"   电场统计:")
            print(f"     最大值: {E_stats['max']:.2e} V/m")
            print(f"     最小值: {E_stats['min']:.2e} V/m")
            print(f"     平均值: {E_stats['mean']:.2e} V/m")
            print(f"     范数: {E_stats['norm']:.2e} V/m")
        
        print("\n3. 验证结果...")
        
        # 简单验证：检查中心点的电势
        center_i = world.nn[0] // 2
        center_j = world.nn[1] // 2  
        center_k = world.nn[2] // 2
        center_phi = world.potential.field[center_i, center_j, center_k, 0]
        center_rho = world.rho.field[center_i+1, center_j+1, center_k+1, 0]
        print(f"   中心点电势: {center_phi:.2e} V")
        print(f"   中心点电荷密度: {center_rho:.2e} C/m³")


        # 对于正电荷分布，中心应该有正电势
        if center_phi > 0:
            print("   ✓ 验证通过：正电荷产生正电势")
        else:
            print("   ⚠ 警告：预期正电势但得到负值")
            
    except Exception as e:
        print(f"   ❌ 求解失败: {e}")
        print("   请确保已安装 FEniCS 库")
        return False
    
    print("\n=== 示例完成 ===")
    return True

if __name__ == "__main__":
    import sys
    
    success = run_poisson_solver_example()
    
    if success:
        print("\n提示：")
        print("1. 如果遇到内存不足，运行: python poisson_solver_example.py --memory-save")
        print("2. 可以修改 create_test_charge_distribution() 函数来测试不同的电荷分布")
        print("3. 可以在 World.potentialSlover() 中修改边界条件")
        print("4. 确保系统已安装 FEniCS 库: pip install fenics")
    else:
        print("\n请检查 FEniCS 安装并重试")
