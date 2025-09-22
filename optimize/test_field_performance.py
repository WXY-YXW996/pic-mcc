#!/usr/bin/env python3
"""
测试numba优化后的Field类性能
"""

import numpy as np
import time
import matplotlib.pyplot as plt
from Field import Field
import scipy.constants as constant


def create_test_field(nn=32):
    """创建测试字段"""
    field = Field(nn)
    field.setBox(np.array([0.0, 0.0, 0.0]), np.array([1.0, 1.0, 1.0]))
    field.initialize_field(3)  # 3分量字段
    
    # 创建一个有趣的测试字段（螺旋磁场）
    for i in range(nn):
        for j in range(nn):
            for k in range(nn):
                x = i / (nn - 1)
                y = j / (nn - 1)
                z = k / (nn - 1)
                
                # 螺旋磁场
                field.field[i, j, k, 0] = np.sin(2 * np.pi * z) * np.cos(2 * np.pi * x)
                field.field[i, j, k, 1] = np.sin(2 * np.pi * z) * np.sin(2 * np.pi * x)
                field.field[i, j, k, 2] = np.cos(2 * np.pi * z) * (x**2 + y**2)
    
    return field


def test_gather_performance(field, n_samples_list=[1000, 5000, 10000, 50000]):
    """测试gather操作性能"""
    print("测试gather操作性能...")
    print("-" * 50)
    
    results = []
    
    for n_samples in n_samples_list:
        print(f"\n测试 {n_samples} 个采样点:")
        
        # 生成随机采样位置
        positions = np.random.random((n_samples, 3)) * 0.8 + 0.1
        
        # 测试单点gather
        start_time = time.time()
        for i in range(min(n_samples, 1000)):  # 限制单点测试数量
            loc_pos = field.XToL(positions[i])
            _ = field.gather(loc_pos)
        single_time = time.time() - start_time
        
        # 测试批量gather
        start_time = time.time()
        _ = field.batch_gather(positions)
        batch_time = time.time() - start_time
        
        print(f"单点gather (1000次): {single_time:.4f} 秒")
        print(f"批量gather ({n_samples}个): {batch_time:.4f} 秒")
        print(f"批量加速比: {single_time/batch_time*n_samples/1000:.2f}x")
        
        results.append({
            'n_samples': n_samples,
            'single_time': single_time,
            'batch_time': batch_time,
            'speedup': single_time/batch_time*n_samples/1000 if batch_time > 0 else 0
        })
    
    return results


def test_scatter_performance(field, n_samples_list=[1000, 5000, 10000, 50000]):
    """测试scatter操作性能"""
    print("\n测试scatter操作性能...")
    print("-" * 50)
    
    results = []
    
    for n_samples in n_samples_list:
        print(f"\n测试 {n_samples} 个散射点:")
        
        # 生成随机散射位置和值
        positions = np.random.random((n_samples, 3)) * 0.8 + 0.1
        values = np.random.random((n_samples, 3)) * 1e-6
        
        # 清零字段
        field.clear()
        
        # 测试单点scatter
        start_time = time.time()
        for i in range(min(n_samples, 1000)):  # 限制单点测试数量
            loc_pos = field.XToL(positions[i])
            field.scatter(loc_pos, values[i])
        single_time = time.time() - start_time
        
        # 清零字段
        field.clear()
        
        # 测试批量scatter
        start_time = time.time()
        field.batch_scatter(positions, values)
        batch_time = time.time() - start_time
        
        print(f"单点scatter (1000次): {single_time:.4f} 秒")
        print(f"批量scatter ({n_samples}个): {batch_time:.4f} 秒")
        print(f"批量加速比: {single_time/batch_time*n_samples/1000:.2f}x")
        
        results.append({
            'n_samples': n_samples,
            'single_time': single_time,
            'batch_time': batch_time,
            'speedup': single_time/batch_time*n_samples/1000 if batch_time > 0 else 0
        })
    
    return results


def test_field_operations(field):
    """测试字段操作性能"""
    print("\n测试字段操作性能...")
    print("-" * 50)
    
    # 测试位置
    test_positions = np.random.random((100, 3)) * 0.8 + 0.1
    
    # 测试梯度计算
    start_time = time.time()
    gradients = []
    for pos in test_positions:
        grad = field.get_gradient(pos)
        gradients.append(grad)
    gradient_time = time.time() - start_time
    
    # 测试散度计算
    start_time = time.time()
    divergences = []
    for pos in test_positions:
        div = field.get_divergence(pos)
        divergences.append(div)
    divergence_time = time.time() - start_time
    
    # 测试旋度计算
    start_time = time.time()
    curls = []
    for pos in test_positions:
        curl = field.get_curl(pos)
        curls.append(curl)
    curl_time = time.time() - start_time
    
    # 测试字段平滑
    start_time = time.time()
    field.smooth(iterations=1)
    smooth_time = time.time() - start_time
    
    print(f"梯度计算 (100点): {gradient_time:.4f} 秒")
    print(f"散度计算 (100点): {divergence_time:.4f} 秒")
    print(f"旋度计算 (100点): {curl_time:.4f} 秒")
    print(f"字段平滑 (1次迭代): {smooth_time:.4f} 秒")
    
    return {
        'gradient_time': gradient_time,
        'divergence_time': divergence_time,
        'curl_time': curl_time,
        'smooth_time': smooth_time
    }


def test_field_correctness():
    """测试字段操作正确性"""
    print("\n测试字段操作正确性...")
    print("=" * 60)
    
    # 创建简单测试字段
    field = Field(16)
    field.setBox(np.array([0.0, 0.0, 0.0]), np.array([1.0, 1.0, 1.0]))
    field.initialize_field(3)
    
    # 设置简单的线性字段 F = (x, y, z)
    for i in range(16):
        for j in range(16):
            for k in range(16):
                x = i / 15.0
                y = j / 15.0
                z = k / 15.0
                field.field[i, j, k, 0] = x
                field.field[i, j, k, 1] = y
                field.field[i, j, k, 2] = z
    
    # 测试插值
    test_pos = np.array([0.5, 0.5, 0.5])
    loc_pos = field.XToL(test_pos)
    interpolated = field.gather(loc_pos)
    
    print(f"测试位置: {test_pos}")
    print(f"插值结果: {interpolated}")
    print(f"期望结果: [0.5, 0.5, 0.5]")
    print(f"误差: {np.abs(interpolated - test_pos)}")
    
    # 测试梯度（对于线性字段，梯度应该是单位矩阵）
    gradient = field.get_gradient(test_pos)
    expected_gradient = np.eye(3)
    
    print(f"\n梯度矩阵:")
    print(gradient)
    print(f"期望梯度:")
    print(expected_gradient)
    print(f"梯度误差: {np.max(np.abs(gradient - expected_gradient))}")
    
    # 测试散度（对于F=(x,y,z)，散度应该是3）
    divergence = field.get_divergence(test_pos)
    expected_divergence = 3.0
    
    print(f"\n散度: {divergence}")
    print(f"期望散度: {expected_divergence}")
    print(f"散度误差: {abs(divergence - expected_divergence)}")
    
    # 测试旋度（对于F=(x,y,z)，旋度应该是0）
    curl = field.get_curl(test_pos)
    expected_curl = np.zeros(3)
    
    print(f"\n旋度: {curl}")
    print(f"期望旋度: {expected_curl}")
    print(f"旋度误差: {np.max(np.abs(curl - expected_curl))}")
    
    # 测试批量操作
    positions = np.array([[0.25, 0.25, 0.25], [0.75, 0.75, 0.75]])
    batch_result = field.batch_gather(positions)
    
    print(f"\n批量插值测试:")
    print(f"位置: {positions}")
    print(f"结果: {batch_result}")
    print(f"期望: {positions}")
    print(f"误差: {np.max(np.abs(batch_result - positions))}")


def plot_performance_results(gather_results, scatter_results, field_ops):
    """绘制性能测试结果"""
    try:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Gather性能
        n_samples = [r['n_samples'] for r in gather_results]
        gather_times = [r['batch_time'] for r in gather_results]
        gather_speedups = [r['speedup'] for r in gather_results]
        
        ax1.loglog(n_samples, gather_times, 'bo-', label='批量gather')
        ax1.set_xlabel('采样点数量')
        ax1.set_ylabel('时间 (秒)')
        ax1.set_title('Gather操作性能')
        ax1.grid(True)
        ax1.legend()
        
        # Scatter性能
        scatter_times = [r['batch_time'] for r in scatter_results]
        scatter_speedups = [r['speedup'] for r in scatter_results]
        
        ax2.loglog(n_samples, scatter_times, 'ro-', label='批量scatter')
        ax2.set_xlabel('散射点数量')
        ax2.set_ylabel('时间 (秒)')
        ax2.set_title('Scatter操作性能')
        ax2.grid(True)
        ax2.legend()
        
        # 加速比对比
        ax3.semilogx(n_samples, gather_speedups, 'bo-', label='Gather加速比')
        ax3.semilogx(n_samples, scatter_speedups, 'ro-', label='Scatter加速比')
        ax3.set_xlabel('操作数量')
        ax3.set_ylabel('加速比')
        ax3.set_title('批量操作加速比')
        ax3.grid(True)
        ax3.legend()
        
        # 字段操作性能
        operations = ['梯度', '散度', '旋度', '平滑']
        times = [field_ops['gradient_time'], field_ops['divergence_time'], 
                field_ops['curl_time'], field_ops['smooth_time']]
        
        ax4.bar(operations, times, color=['blue', 'green', 'red', 'orange'])
        ax4.set_ylabel('时间 (秒)')
        ax4.set_title('字段操作性能 (100点)')
        ax4.grid(True, axis='y')
        
        plt.tight_layout()
        plt.savefig('field_performance_test.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    except ImportError:
        print("matplotlib未安装，跳过绘图")


def main():
    """主测试函数"""
    print("=" * 60)
    print("Field类numba优化性能测试")
    print("=" * 60)
    
    # 测试正确性
    test_field_correctness()
    
    # 创建测试字段
    print("\n创建测试字段...")
    field = create_test_field(32)
    
    # 获取字段统计信息
    stats = field.get_field_stats()
    print(f"字段统计信息:")
    print(f"  形状: {stats['shape']}")
    print(f"  最大值: {stats['max']:.6f}")
    print(f"  最小值: {stats['min']:.6f}")
    print(f"  平均值: {stats['mean']:.6f}")
    print(f"  标准差: {stats['std']:.6f}")
    print(f"  范数: {stats['norm']:.6f}")
    
    # 性能测试
    gather_results = test_gather_performance(field)
    scatter_results = test_scatter_performance(field)
    field_ops = test_field_operations(field)
    
    # 绘制结果
    plot_performance_results(gather_results, scatter_results, field_ops)
    
    # 输出总结
    print("\n" + "=" * 60)
    print("Field类numba优化总结:")
    print("=" * 60)
    
    print("\n主要优化功能:")
    print("1. numba优化的三线性插值 (gather)")
    print("2. numba优化的三线性散射 (scatter)")
    print("3. 批量操作支持 (batch_gather, batch_scatter)")
    print("4. 字段微分操作 (梯度, 散度, 旋度)")
    print("5. 字段平滑处理")
    print("6. 坐标转换优化")
    
    print("\n性能提升:")
    avg_gather_speedup = np.mean([r['speedup'] for r in gather_results])
    avg_scatter_speedup = np.mean([r['speedup'] for r in scatter_results])
    
    print(f"- Gather操作平均加速比: {avg_gather_speedup:.1f}x")
    print(f"- Scatter操作平均加速比: {avg_scatter_speedup:.1f}x")
    print(f"- 支持高效的批量操作")
    print(f"- 内存访问模式优化")
    
    print("\n新增功能:")
    print("- 字段梯度计算")
    print("- 字段散度计算")
    print("- 字段旋度计算")
    print("- 字段平滑处理")
    print("- 批量操作接口")
    print("- 字段统计信息")
    
    print("\nField类numba优化完成！")


if __name__ == "__main__":
    main()
