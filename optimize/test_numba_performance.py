#!/usr/bin/env python3
"""
测试numba优化后的Species类性能
"""

import numpy as np
import time
import matplotlib.pyplot as plt
from Species import Species
import World
import Field
import scipy.constants as constant

def create_test_environment():
    """创建测试环境"""
    # 创建世界
    world = World.World(32)  # 32x32x32网格
    world.setBox(np.array([0.0, 0.0, 0.0]), np.array([1.0, 1.0, 1.0]))
    world.setTime(1e-12, 1000)  # 1ps时间步长，1000步
    
    # 创建场
    b_field = Field.Field(32)
    b_field.setBox(np.array([0.0, 0.0, 0.0]), np.array([1.0, 1.0, 1.0]))
    b_field.field = np.random.random((32, 32, 32, 3)) * 0.1  # 随机磁场
    
    e_field = Field.Field(32)
    e_field.setBox(np.array([0.0, 0.0, 0.0]), np.array([1.0, 1.0, 1.0]))
    e_field.field = np.random.random((32, 32, 32, 3)) * 1e6  # 随机电场
    
    return world, b_field, e_field

def test_performance(n_particles_list=[100, 500, 1000, 5000]):
    """测试不同粒子数量下的性能"""
    
    print("开始性能测试...")
    print("=" * 60)
    
    results = []
    
    for n_particles in n_particles_list:
        print(f"\n测试 {n_particles} 个粒子:")
        print("-" * 40)
        
        # 创建测试环境
        world, b_field, e_field = create_test_environment()
        
        # 创建电子物种
        electrons = Species("electrons", constant.m_e, -constant.e, b_field, e_field, world)
        
        # 添加粒子
        print("添加粒子...")
        start_time = time.time()
        
        for i in range(n_particles):
            pos = np.random.random(3) * 0.8 + 0.1  # 在0.1-0.9范围内随机位置
            vel = (np.random.random(3) - 0.5) * 1e7  # 随机速度
            electrons.addParticles(pos, vel)
        
        add_time = time.time() - start_time
        print(f"添加粒子耗时: {add_time:.4f} 秒")
        
        # 测试推进器性能
        print("测试Boris推进器...")
        n_steps = 100
        
        start_time = time.time()
        for step in range(n_steps):
            electrons.advance()
            electrons.boundary()
        advance_time = time.time() - start_time
        
        print(f"推进 {n_steps} 步耗时: {advance_time:.4f} 秒")
        print(f"平均每步耗时: {advance_time/n_steps:.6f} 秒")
        print(f"每个粒子每步耗时: {advance_time/(n_steps*n_particles)*1e6:.3f} 微秒")
        
        # 测试动能和动量计算
        print("测试动能和动量计算...")
        
        start_time = time.time()
        kinetic = electrons.getKinetic()
        kinetic_time = time.time() - start_time
        
        start_time = time.time()
        momentum = electrons.getMomentum()
        momentum_time = time.time() - start_time
        
        print(f"动能计算耗时: {kinetic_time:.6f} 秒")
        print(f"动量计算耗时: {momentum_time:.6f} 秒")
        
        # 统计信息
        avg_kinetic = np.mean(kinetic)
        # avg_speed = np.mean(np.linalg.norm(electrons.particles.es, axis=1))
        
        print(f"平均动能: {avg_kinetic:.2e} J")
        # print(f"平均速度: {avg_speed:.2e} m/s")
        
        results.append({
            'n_particles': n_particles,
            'add_time': add_time,
            'advance_time': advance_time,
            'kinetic_time': kinetic_time,
            'momentum_time': momentum_time,
            'avg_kinetic': avg_kinetic,
            # 'avg_speed': avg_speed
        })
    
    return results

def plot_performance_results(results):
    """绘制性能测试结果"""
    n_particles = [r['n_particles'] for r in results]
    advance_times = [r['advance_time'] for r in results]
    kinetic_times = [r['kinetic_time'] for r in results]
    momentum_times = [r['momentum_time'] for r in results]
    
    plt.figure(figsize=(12, 8))
    
    # 推进器性能
    plt.subplot(2, 2, 1)
    plt.loglog(n_particles, advance_times, 'bo-', label='Boris推进器')
    plt.xlabel('粒子数量')
    plt.ylabel('时间 (秒)')
    plt.title('Boris推进器性能 (100步)')
    plt.grid(True)
    plt.legend()
    
    # 动能计算性能
    plt.subplot(2, 2, 2)
    plt.loglog(n_particles, kinetic_times, 'ro-', label='动能计算')
    plt.xlabel('粒子数量')
    plt.ylabel('时间 (秒)')
    plt.title('动能计算性能')
    plt.grid(True)
    plt.legend()
    
    # 动量计算性能
    plt.subplot(2, 2, 3)
    plt.loglog(n_particles, momentum_times, 'go-', label='动量计算')
    plt.xlabel('粒子数量')
    plt.ylabel('时间 (秒)')
    plt.title('动量计算性能')
    plt.grid(True)
    plt.legend()
    
    # 综合性能对比
    plt.subplot(2, 2, 4)
    plt.loglog(n_particles, advance_times, 'bo-', label='Boris推进器')
    plt.loglog(n_particles, kinetic_times, 'ro-', label='动能计算')
    plt.loglog(n_particles, momentum_times, 'go-', label='动量计算')
    plt.xlabel('粒子数量')
    plt.ylabel('时间 (秒)')
    plt.title('性能对比')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('numba_performance_test.png', dpi=300, bbox_inches='tight')
    plt.show()

def test_correctness():
    """测试计算正确性"""
    print("\n" + "=" * 60)
    print("测试计算正确性...")
    print("=" * 60)
    
    # 创建简单测试环境
    world, b_field, e_field = create_test_environment()
    
    # 创建电子
    electrons = Species("electrons", constant.m_e, -constant.e, b_field, e_field, world)
    
    # 添加一个测试粒子
    pos = np.array([0.5, 0.5, 0.5])
    vel = np.array([1e6, 0.0, 0.0])  # 1 MeV电子
    electrons.addParticles(pos, vel)
    
    print(f"初始位置: {pos}")
    print(f"初始速度: {vel}")
    
    # 推进几步
    for i in range(5):
        electrons.advance()
        print(f"步骤 {i+1}:")
        print(f"  位置: {electrons.particles[0].pos}")
        print(f"  速度: {electrons.particles[0].vel}")
        
        # 计算动能
        kinetic = electrons.getKinetic()
        print(f"  动能: {kinetic[0]:.2e} J")
        
        # 计算动量
        momentum = electrons.getMomentum()
        print(f"  动量: {momentum[0]}")

if __name__ == "__main__":
    # 测试正确性
    test_correctness()
    
    # 性能测试
    results = test_performance([100, 500, 1000, 2000])
    
    # 绘制结果
    try:
        plot_performance_results(results)
    except ImportError:
        print("\n注意: matplotlib未安装，跳过绘图")
    
    # 输出总结
    print("\n" + "=" * 60)
    print("性能测试总结:")
    print("=" * 60)
    
    for result in results:
        n = result['n_particles']
        t = result['advance_time']
        print(f"{n:4d} 粒子: {t:.4f}s (100步), {t/(100*n)*1e6:.2f} μs/粒子/步")
    
    print("\nnumba优化完成！主要改进:")
    print("1. 使用numba.njit装饰器优化核心计算函数")
    print("2. 将粒子数据存储为连续的numpy数组")
    print("3. 优化了Boris推进器、边界条件和物理量计算")
    print("4. 保持了与原始接口的兼容性")
