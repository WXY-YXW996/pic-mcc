"""
PIC-MCC 仿真诊断模块
提供各种物理量的计算和分析功能
包括能量诊断、密度诊断、分布函数诊断等
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.spatial.distance import pdist, squareform
import h5py
from mpi4py import MPI
import time

class PhysicalDiagnostics:
    """
    物理诊断类
    计算各种物理量和统计量
    """
    
    def __init__(self, world):
        """
        初始化诊断器
        
        参数:
            world: World对象
        """
        self.world = world
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        
        # 历史数据存储
        self.history = {
            'time': [],
            'kinetic_energy': [],
            'potential_energy': [],
            'total_energy': [],
            'momentum': [],
            'particle_counts': [],
            'density_max': [],
            'temperature': [],
            'pressure': []
        }
    
    def compute_kinetic_energy(self, species_list):
        """
        计算系统总动能和各物种动能
        
        参数:
            species_list: list - 粒子种类列表
            
        返回:
            dict - 包含总动能和各物种动能的字典
        """
        energies = {'total': 0.0, 'species': []}
        
        for species in species_list:
            if species.npar > 0:
                # 计算非相对论动能
                vel_squared = np.sum(species.vel**2, axis=1)
                kinetic = 0.5 * species.mass * np.sum(vel_squared)
                
                # MPI归约
                total_kinetic = self.comm.allreduce(kinetic, op=MPI.SUM)
                energies['species'].append(total_kinetic)
                energies['total'] += total_kinetic
            else:
                energies['species'].append(0.0)
        
        return energies
    
    def compute_momentum(self, species_list):
        """
        计算系统总动量
        
        参数:
            species_list: list - 粒子种类列表
            
        返回:
            dict - 包含总动量和各分量的字典
        """
        total_momentum = np.zeros(3)
        species_momentum = []
        
        for species in species_list:
            if species.npar > 0:
                momentum = species.mass * np.sum(species.vel, axis=0)
                
                # MPI归约
                global_momentum = np.zeros(3)
                self.comm.Allreduce(momentum, global_momentum, op=MPI.SUM)
                species_momentum.append(global_momentum)
                total_momentum += global_momentum
            else:
                species_momentum.append(np.zeros(3))
        
        return {
            'total': total_momentum,
            'magnitude': np.linalg.norm(total_momentum),
            'species': species_momentum
        }
    
    def compute_temperature(self, species_list):
        """
        计算各物种的温度（基于速度分布）
        
        参数:
            species_list: list - 粒子种类列表
            
        返回:
            list - 各物种的温度
        """
        temperatures = []
        
        for species in species_list:
            if species.npar > 0:
                # 计算平均速度
                mean_vel = np.mean(species.vel, axis=0)
                
                # 计算速度方差
                vel_variance = np.var(species.vel, axis=0)
                
                # 温度 = m * <v^2> / (3 * k_B)
                # 这里我们使用简化的温度定义，不除以玻尔兹曼常数
                temp = species.mass * np.mean(vel_variance) / 3.0
                
                # MPI归约获取全局温度
                global_temp = self.comm.allreduce(temp, op=MPI.SUM) / self.comm.Get_size()
                temperatures.append(global_temp)
            else:
                temperatures.append(0.0)
        
        return temperatures
    
    def compute_density_statistics(self, species_list):
        """
        计算密度统计量
        
        参数:
            species_list: list - 粒子种类列表
            
        返回:
            dict - 密度统计信息
        """
        density_stats = {'species': []}
        
        for species in species_list:
            if hasattr(species, 'den') and species.den.field is not None:
                density = species.den.field
                
                stats = {
                    'max': np.max(density),
                    'min': np.min(density),
                    'mean': np.mean(density),
                    'std': np.std(density),
                    'total': np.sum(density) * species.den.volume[0,0,0,0]  # 总粒子数近似
                }
                
                # MPI归约
                for key in ['max', 'min', 'mean', 'std', 'total']:
                    if key == 'max':
                        stats[key] = self.comm.allreduce(stats[key], op=MPI.MAX)
                    elif key == 'min':
                        stats[key] = self.comm.allreduce(stats[key], op=MPI.MIN)
                    else:
                        stats[key] = self.comm.allreduce(stats[key], op=MPI.SUM)
                        if key in ['mean', 'std']:
                            stats[key] /= self.comm.Get_size()
                
                density_stats['species'].append(stats)
            else:
                density_stats['species'].append({'max': 0, 'min': 0, 'mean': 0, 'std': 0, 'total': 0})
        
        return density_stats
    
    def compute_field_energy(self, fields):
        """
        计算场能量
        
        参数:
            fields: dict - 场字典
            
        返回:
            dict - 各场的能量
        """
        field_energies = {}
        
        for field_name, field_obj in fields.items():
            if field_obj.field is not None:
                # 计算场能量密度
                if len(field_obj.field.shape) == 4:  # 矢量场
                    energy_density = 0.5 * np.sum(field_obj.field**2, axis=3)
                else:  # 标量场
                    energy_density = 0.5 * field_obj.field**2
                
                # 积分得到总能量
                total_energy = np.sum(energy_density) * field_obj.volume[0,0,0,0]
                
                # MPI归约
                global_energy = self.comm.allreduce(total_energy, op=MPI.SUM)
                field_energies[field_name] = global_energy
            else:
                field_energies[field_name] = 0.0
        
        return field_energies
    
    def compute_velocity_distribution(self, species, bins=50, velocity_range=None):
        """
        计算速度分布函数
        
        参数:
            species: Species对象
            bins: int - 直方图bin数
            velocity_range: tuple - 速度范围
            
        返回:
            dict - 速度分布信息
        """
        if species.npar == 0:
            return {'v_bins': np.array([]), 'distribution': np.array([]), 
                   'vx_dist': np.array([]), 'vy_dist': np.array([]), 'vz_dist': np.array([])}
        
        # 计算速度大小
        vel_magnitude = np.linalg.norm(species.vel, axis=1)
        
        # 确定速度范围
        if velocity_range is None:
            v_min, v_max = 0, np.max(vel_magnitude) * 1.1
        else:
            v_min, v_max = velocity_range
        
        # 计算速度分布
        hist, bin_edges = np.histogram(vel_magnitude, bins=bins, range=(v_min, v_max))
        v_bins = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # 归一化
        distribution = hist / (np.sum(hist) * (bin_edges[1] - bin_edges[0]))
        
        # 计算各分量的分布
        vx_hist, _ = np.histogram(species.vel[:, 0], bins=bins)
        vy_hist, _ = np.histogram(species.vel[:, 1], bins=bins)
        vz_hist, _ = np.histogram(species.vel[:, 2], bins=bins)
        
        return {
            'v_bins': v_bins,
            'distribution': distribution,
            'vx_dist': vx_hist,
            'vy_dist': vy_hist,
            'vz_dist': vz_hist,
            'v_mean': np.mean(vel_magnitude),
            'v_std': np.std(vel_magnitude)
        }
    
    def compute_spatial_correlation(self, species, max_distance=None, bins=50):
        """
        计算粒子空间相关函数
        
        参数:
            species: Species对象
            max_distance: float - 最大距离
            bins: int - 距离bin数
            
        返回:
            dict - 相关函数信息
        """
        if species.npar < 2:
            return {'r_bins': np.array([]), 'correlation': np.array([])}
        
        # 计算粒子间距离（子采样以避免内存问题）
        max_particles = min(1000, species.npar)  # 限制粒子数
        indices = np.random.choice(species.npar, max_particles, replace=False)
        positions = species.pos[indices]
        
        # 计算距离矩阵
        distances = pdist(positions)
        
        # 确定距离范围
        if max_distance is None:
            max_distance = np.max(distances)
        
        # 计算径向分布函数
        hist, bin_edges = np.histogram(distances, bins=bins, 
                                     range=(0, max_distance))
        r_bins = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # 归一化（简化版本）
        correlation = hist / np.max(hist) if np.max(hist) > 0 else hist
        
        return {
            'r_bins': r_bins,
            'correlation': correlation,
            'mean_distance': np.mean(distances),
            'std_distance': np.std(distances)
        }
    
    def update_history(self, species_list, fields=None):
        """
        更新历史数据
        
        参数:
            species_list: list - 粒子种类列表
            fields: dict - 场字典
        """
        # 更新时间
        self.history['time'].append(self.world.time)
        
        # 更新能量
        kinetic = self.compute_kinetic_energy(species_list)
        self.history['kinetic_energy'].append(kinetic['total'])
        
        if fields:
            field_energy = self.compute_field_energy(fields)
            potential = sum(field_energy.values())
            self.history['potential_energy'].append(potential)
            self.history['total_energy'].append(kinetic['total'] + potential)
        else:
            self.history['potential_energy'].append(0.0)
            self.history['total_energy'].append(kinetic['total'])
        
        # 更新动量
        momentum = self.compute_momentum(species_list)
        self.history['momentum'].append(momentum['magnitude'])
        
        # 更新粒子数
        particle_counts = [species.npar for species in species_list]
        global_counts = []
        for count in particle_counts:
            global_count = self.comm.allreduce(count, op=MPI.SUM)
            global_counts.append(global_count)
        self.history['particle_counts'].append(global_counts)
        
        # 更新密度统计
        density_stats = self.compute_density_statistics(species_list)
        max_density = max([stats['max'] for stats in density_stats['species']] + [0])
        self.history['density_max'].append(max_density)
        
        # 更新温度
        temperatures = self.compute_temperature(species_list)
        self.history['temperature'].append(temperatures)
    
    def save_diagnostics(self, filename, species_list, fields=None):
        """
        保存诊断数据到文件
        
        参数:
            filename: str - 文件名
            species_list: list - 粒子种类列表
            fields: dict - 场字典
        """
        if self.rank != 0:
            return
        
        with h5py.File(filename, 'w') as f:
            # 保存历史数据
            history_group = f.create_group('history')
            for key, value in self.history.items():
                if value:  # 只保存非空数据
                    history_group.create_dataset(key, data=np.array(value))
            
            # 保存当前状态诊断
            current_group = f.create_group('current_state')
            
            # 动能诊断
            kinetic = self.compute_kinetic_energy(species_list)
            kinetic_group = current_group.create_group('kinetic_energy')
            kinetic_group.create_dataset('total', data=kinetic['total'])
            kinetic_group.create_dataset('species', data=kinetic['species'])
            
            # 动量诊断
            momentum = self.compute_momentum(species_list)
            momentum_group = current_group.create_group('momentum')
            momentum_group.create_dataset('total', data=momentum['total'])
            momentum_group.create_dataset('magnitude', data=momentum['magnitude'])
            momentum_group.create_dataset('species', data=np.array(momentum['species']))
            
            # 温度诊断
            temperatures = self.compute_temperature(species_list)
            current_group.create_dataset('temperature', data=temperatures)
            
            # 密度诊断
            density_stats = self.compute_density_statistics(species_list)
            density_group = current_group.create_group('density_statistics')
            for i, stats in enumerate(density_stats['species']):
                species_group = density_group.create_group(f'species_{i}')
                for key, value in stats.items():
                    species_group.create_dataset(key, data=value)
            
            # 场能量诊断
            if fields:
                field_energies = self.compute_field_energy(fields)
                field_group = current_group.create_group('field_energy')
                for field_name, energy in field_energies.items():
                    field_group.create_dataset(field_name, data=energy)
            
            # 速度分布诊断
            velocity_group = current_group.create_group('velocity_distributions')
            for i, species in enumerate(species_list):
                vel_dist = self.compute_velocity_distribution(species)
                species_vel_group = velocity_group.create_group(f'species_{i}')
                for key, value in vel_dist.items():
                    if len(value) > 0:
                        species_vel_group.create_dataset(key, data=value)
            
            # 元数据
            f.attrs['timestamp'] = time.time()
            f.attrs['simulation_time'] = self.world.time
            f.attrs['timestep'] = self.world.ts
            f.attrs['total_species'] = len(species_list)
        
        print(f"诊断数据已保存到: {filename}")


class RealTimeDiagnostics:
    """
    实时诊断显示类
    """
    
    def __init__(self, world, update_interval=10):
        """
        初始化实时诊断
        
        参数:
            world: World对象
            update_interval: int - 更新间隔
        """
        self.world = world
        self.update_interval = update_interval
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        
        # 数据缓存
        self.energy_buffer = []
        self.momentum_buffer = []
        self.particle_buffer = []
        self.time_buffer = []
        
        if self.rank == 0:
            plt.ion()
            self.fig, self.axes = plt.subplots(2, 2, figsize=(12, 8))
            self.fig.suptitle('PIC-MCC 实时诊断')
    
    def update(self, diagnostics, species_list):
        """
        更新实时诊断显示
        
        参数:
            diagnostics: PhysicalDiagnostics对象
            species_list: list - 粒子种类列表
        """
        if self.rank != 0 or self.world.ts % self.update_interval != 0:
            return
        
        # 更新数据缓存
        if len(diagnostics.history['time']) > 0:
            self.time_buffer = diagnostics.history['time'][-100:]  # 保留最近100个点
            self.energy_buffer = diagnostics.history['total_energy'][-100:]
            self.momentum_buffer = diagnostics.history['momentum'][-100:]
            
            if diagnostics.history['particle_counts']:
                self.particle_buffer = diagnostics.history['particle_counts'][-100:]
        
        # 清除图形
        for ax in self.axes.flat:
            ax.clear()
        
        # 绘制能量演化
        if self.energy_buffer:
            self.axes[0, 0].plot(self.time_buffer, self.energy_buffer, 'b-')
            self.axes[0, 0].set_xlabel('时间 (s)')
            self.axes[0, 0].set_ylabel('总能量 (J)')
            self.axes[0, 0].set_title('能量演化')
            self.axes[0, 0].grid(True)
        
        # 绘制动量演化
        if self.momentum_buffer:
            self.axes[0, 1].plot(self.time_buffer, self.momentum_buffer, 'r-')
            self.axes[0, 1].set_xlabel('时间 (s)')
            self.axes[0, 1].set_ylabel('动量大小 (kg⋅m/s)')
            self.axes[0, 1].set_title('动量演化')
            self.axes[0, 1].grid(True)
        
        # 绘制粒子数演化
        if self.particle_buffer:
            particle_array = np.array(self.particle_buffer)
            for i in range(particle_array.shape[1]):
                self.axes[1, 0].plot(self.time_buffer, particle_array[:, i], 
                                   label=f'Species {i}')
            self.axes[1, 0].set_xlabel('时间 (s)')
            self.axes[1, 0].set_ylabel('粒子数')
            self.axes[1, 0].set_title('粒子数演化')
            self.axes[1, 0].legend()
            self.axes[1, 0].grid(True)
        
        # 绘制当前速度分布
        if len(species_list) > 0 and species_list[0].npar > 0:
            vel_dist = diagnostics.compute_velocity_distribution(species_list[0])
            if len(vel_dist['v_bins']) > 0:
                self.axes[1, 1].plot(vel_dist['v_bins'], vel_dist['distribution'])
                self.axes[1, 1].set_xlabel('速度大小 (m/s)')
                self.axes[1, 1].set_ylabel('分布函数')
                self.axes[1, 1].set_title('速度分布')
                self.axes[1, 1].set_yscale('log')
                self.axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.draw()
        plt.pause(0.01)
    
    def close(self):
        """
        关闭实时诊断
        """
        if self.rank == 0:
            plt.ioff()
            plt.close(self.fig)
