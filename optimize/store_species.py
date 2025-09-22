import numpy as np
import numba
from numba import types
from numba.typed import Dict

import World
import Field
import scipy.constants as constant


# 定义numba兼容的粒子数据结构
@numba.experimental.jitclass([
    ('pos', numba.float64[:]),
    ('vel', numba.float64[:])
])
class ParticleData:
    def __init__(self, pos, vel):
        self.pos = pos
        self.vel = vel


# numba优化的字段插值函数
@numba.njit
def gather_field(field_data, loc_pos):
    """
    numba优化的字段插值函数
    
    参数:
        field_data: np.ndarray - 字段数据
        loc_pos: np.ndarray - 局部坐标位置
    
    返回:
        np.ndarray - 插值后的字段值
    """
    i = int(loc_pos[0])
    di = loc_pos[0] - i
    
    j = int(loc_pos[1])
    dj = loc_pos[1] - j
    
    k = int(loc_pos[2])
    dk = loc_pos[2] - k
    
    # 确保索引在有效范围内
    ni, nj, nk = field_data.shape[0], field_data.shape[1], field_data.shape[2]
    i = max(0, min(i, ni-2))
    j = max(0, min(j, nj-2))
    k = max(0, min(k, nk-2))
    
    val = field_data[i,j,k] * (1 - di) * (1 - dj) * (1 - dk) + \
          field_data[i + 1,j,k] * (di) * (1 - dj) * (1 - dk) + \
          field_data[i + 1,j + 1,k] * (di) * (dj) * (1 - dk) + \
          field_data[i,j + 1,k] * (1 - di) * (dj) * (1 - dk) + \
          field_data[i,j,k + 1] * (1 - di) * (1 - dj) * (dk) + \
          field_data[i + 1,j,k + 1] * (di) * (1 - dj) * (dk) + \
          field_data[i + 1,j + 1,k + 1] * (di) * (dj) * (dk) + \
          field_data[i,j + 1,k + 1] * (1 - di) * (dj) * (dk)
    return val


@numba.njit
def pos_to_local(pos, box_min, dh):
    """
    将物理坐标转换为局部网格坐标
    
    参数:
        pos: np.ndarray - 物理位置
        box_min: np.ndarray - 网格最小边界
        dh: np.ndarray - 网格间距
    
    返回:
        np.ndarray - 局部网格坐标
    """
    return (pos - box_min) / dh


@numba.njit
def boris_advance_particles(positions, velocities, e_field, b_field, 
                          box_min, dh, charge, mass, dt):
    """
    numba优化的Boris推进器
    
    参数:
        positions: np.ndarray - 粒子位置数组 (N, 3)
        velocities: np.ndarray - 粒子速度数组 (N, 3)
        e_field: np.ndarray - 电场数据
        b_field: np.ndarray - 磁场数据
        box_min: np.ndarray - 网格最小边界
        dh: np.ndarray - 网格间距
        charge: float - 粒子电荷
        mass: float - 粒子质量
        dt: float - 时间步长
    """
    npar = positions.shape[0]
    
    for i in range(npar):
        # 获取粒子位置的局部坐标
        lc_pos = pos_to_local(positions[i], box_min, dh)
        
        # 插值获取电场和磁场
        e_part = gather_field(e_field, lc_pos)
        b_part = gather_field(b_field, lc_pos)
        
        # Boris推进器算法
        ff = charge / mass * dt / 2.0
        
        v_minus = velocities[i] + ff * e_part
        
        t = ff * b_part
        
        v_prime = v_minus + np.cross(v_minus, t)
        
        t_dot_t = np.dot(t, t)
        s = 2.0 * t / (1.0 + t_dot_t)
        
        v_plus = v_minus + np.cross(v_prime, s)
        
        velocities[i] = v_plus + ff * e_part
        
        positions[i] += velocities[i] * dt


@numba.njit
def apply_boundary_conditions(positions, velocities, box_min, box_max):
    """
    numba优化的边界条件处理
    
    参数:
        positions: np.ndarray - 粒子位置数组
        velocities: np.ndarray - 粒子速度数组
        box_min: np.ndarray - 最小边界
        box_max: np.ndarray - 最大边界
    """
    npar = positions.shape[0]
    
    for i in range(npar):
        for j in range(3):
            if positions[i, j] < box_min[j]:
                positions[i, j] = box_min[j]
                velocities[i, j] = -velocities[i, j]
            elif positions[i, j] > box_max[j]:
                positions[i, j] = box_max[j]
                velocities[i, j] = -velocities[i, j]


@numba.njit
def compute_kinetic_energy(velocities, mass):
    """
    numba优化的动能计算
    
    参数:
        velocities: np.ndarray - 粒子速度数组
        mass: float - 粒子质量
    
    返回:
        np.ndarray - 粒子动能数组
    """
    npar = velocities.shape[0]
    kinetic = np.zeros(npar)
    c = constant.c
    
    for i in range(npar):
        beta = velocities[i] / c
        beta_squared = np.dot(beta, beta)
        gamma = 1.0 / np.sqrt(1.0 - beta_squared)
        kinetic[i] = mass * c * c * (gamma - 1.0)
    
    return kinetic


@numba.njit
def compute_momentum(velocities, mass):
    """
    numba优化的动量计算
    
    参数:
        velocities: np.ndarray - 粒子速度数组
        mass: float - 粒子质量
    
    返回:
        np.ndarray - 粒子动量数组
    """
    npar = velocities.shape[0]
    momentum = np.zeros((npar, 3))
    c = constant.c
    
    for i in range(npar):
        beta = velocities[i] / c
        beta_squared = np.dot(beta, beta)
        gamma = 1.0 / np.sqrt(1.0 - beta_squared)
        momentum[i] = mass * gamma * velocities[i]
    
    return momentum


class Species:
    def __init__(self, species_name: str, mass: float, charge: float, 
                 b: Field.Field, e: Field.Field, world: World.World):
        """
        初始化粒子种类对象
        
        参数:
            species_name: str - 粒子种类名称
            mass: float - 粒子质量
            charge: float - 粒子电荷
            b: Field.Field - 磁场对象
            e: Field.Field - 电场对象
            world: World.World - 求解域对象
        """
        self.name = species_name
        self.pos = np.empty((0, 3), dtype=np.float64)
        self.vel = np.empty((0, 3), dtype=np.float64)
        self.mass = mass
        self.charge = charge
        self.b = b
        self.e = e
        self.world = world
        self.npar = 0        
        
        
    def addParticles(self, pos: np.ndarray, vel: np.ndarray) -> None:
        """
        添加粒子到物种中
        
        参数:
            pos: np.ndarray - 粒子位置坐标数组
            vel: np.ndarray - 粒子速度向量数组
        """
        if not self.world.inbound(pos):
            raise ValueError("添加粒子失败：位置超出边界！")
        
        self.npar += 1

        self.pos = np.vstack([self.pos, pos.reshape(1, -1)])
        self.vel = np.vstack([self.vel, vel.reshape(1, -1)])
    
    def advance(self) -> None:
        """
        Boris推进器推进所有粒子（numba优化版本）
        参考资料: Birdsall, C. K., & Langdon, A. B. (2004). Plasma physics via computer simulation. CRC Press.
        """
        if self.npar == 0:
            return
            
        # 使用numba优化的推进器
        boris_advance_particles(
            self.pos,
            self.vel,
            self.e.field if hasattr(self.e, 'field') else np.zeros((10, 10, 10, 3)),
            self.b.field if hasattr(self.b, 'field') else np.zeros((10, 10, 10, 3)),
            self.world.box_min,
            self.world.dh,
            self.charge,
            self.mass,
            self.world.dt
        )
            
    def boundary(self) -> None:
        """
        处理粒子边界条件，简单反弹边界（numba优化版本）
        """
        if self.npar == 0:
            return
            
        apply_boundary_conditions(
            self.pos,
            self.vel,
            self.world.box_min,
            self.world.box_max
        )
    
    def getKinetic(self):
        """
        计算所有粒子的动能（numba优化版本）
        
        返回: np.ndarray - 粒子动能数组
        """
        if self.npar == 0:
            return np.array([])
            
        kinetic = compute_kinetic_energy(self.velocities, self.mass)
        self.kinetic = kinetic
        return kinetic
    
    def getMomentum(self):
        """
        计算所有粒子的动量（numba优化版本）
        
        返回: np.ndarray - 粒子动量数组
        """
        if self.npar == 0:
            return np.empty((0, 3))
            
        momentum = compute_momentum(self.velocities, self.mass)
        self.momentum = momentum
        return momentum
