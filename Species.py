import numpy as np
import numba
from numba import types, njit, prange
from numba.typed import Dict

import World
import Field
import scipy.constants as constant



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
def boris_advance_particles(positions, velocities,e_part, b_part, charge, mass, dt):
    """
    numba优化的Boris推进器
    
    参数:
        positions: np.ndarray - 粒子位置数组 (N, 3)
        velocities: np.ndarray - 粒子速度数组 (N, 3)
        e_part: np.ndarray - 粒子处的电场数组 (N, 3)
        b_part: np.ndarray - 粒子处的磁场数组 (N, 3)
        charge: float - 粒子电荷
        mass: float - 粒子质量
        dt: float - 时间步长
    """
    npar = positions.shape[0]
    
    for i in range(npar):

        # Boris推进器算法
        ff = charge / mass * dt / 2.0
        
        v_minus = velocities[i] + ff * e_part[i]
        
        t = ff * b_part[i]
        
        v_prime = v_minus + np.cross(v_minus, t)
        
        t_dot_t = np.dot(t, t)
        s = 2.0 * t / (1.0 + t_dot_t)
        
        v_plus = v_minus + np.cross(v_prime, s)
        
        velocities[i] = v_plus + ff * e_part[i]
        
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
def compute_kinetic_energy_particles(velocities, mass):
    """
    计算粒子动能
    """
    c = constant.c

    beta = velocities / c
    beta_squared = np.dot(beta, beta)
    gamma = 1.0 / np.sqrt(1.0 - beta_squared)
    kinetic = mass * c * c * (gamma - 1.0) / constant.electron_volt

    return kinetic


@numba.njit
def compute_kinetic_energy(velocities, kinetic, mass):
    """
    计算粒子动能
    """
    npar = velocities.shape[0]

    for i in range(npar):
        kinetic[i] = compute_kinetic_energy_particles(velocities[i], mass)


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
    def __init__(self, species_name: str, mass: float, charge: float, weight: float,
                 b: Field.Field, e_r: Field.Field, e_i: Field.Field, world: World.World):
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
        self.mass = mass
        self.charge = charge
        self.weight = weight

        self.pos = np.empty((0, 3), dtype=np.float64)
        self.vel = np.empty((0, 3), dtype=np.float64)
        self.kinetic = np.empty((0, 1), dtype=np.float64)

        
        self.b = b
        self.e_r = e_r
        self.e_i = e_i
        self.world = world
        self.npar = 0
        
        self.den = Field.Field(self.world.nn)
        self.den.setBox(self.world.box_min, self.world.box_max)
        self.den.initializeField(n_components=1)
        
        
        
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
        kinetic = compute_kinetic_energy_particles(vel, self.mass)
        self.kinetic = np.vstack([self.kinetic, np.array([[kinetic]])])


    
    def advance(self) -> None:
        """
        Boris推进器推进所有粒子（numba优化版本）
        参考资料: Birdsall, C. K., & Langdon, A. B. (2004). Plasma physics via computer simulation. CRC Press.
        """
        if self.npar == 0:
            return
        
        # 得到磁场
        self.b_part = self.b.batchGather(self.pos)

        # 得到电场
        real_time = self.world.time
        fre = self.world.fre
        e_r_part = self.e_r.batchGather(self.pos)
        e_i_part = self.e_i.batchGather(self.pos)

        cosphs = np.cos(fre * np.pi * 2 * real_time)
        sinphs = np.sin(fre * np.pi * 2 * real_time)
        self.e_part = e_r_part * cosphs - e_i_part * sinphs

        # self.e_part = self.e.batchGather(self.pos)
        
        boris_advance_particles(
            self.pos,
            self.vel,
            self.e_part,
            self.b_part,
            self.charge,
            self.mass,
            self.world.dt
        )
        self.getKinetic()
            
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
            
        kinetic = compute_kinetic_energy(self.vel, self.kinetic, self.mass)
        return kinetic
    
    def getMomentum(self):
        """
        计算所有粒子的动量（numba优化版本）
        
        返回: np.ndarray - 粒子动量数组
        """
        if self.npar == 0:
            return np.empty((0, 3))
            
        momentum = compute_momentum(self.vel, self.mass)
        self.momentum = momentum
        return momentum
    
    def computeNumberDensity(self):
        """
        计算粒子数密度

        返回: np.ndarray - 粒子数密度数组
        """
        if self.npar == 0:
            return np.zeros(self.world.nn, dtype=np.float64)
        
        self.den.initializeField(n_components=1)
        vals = np.ones((self.npar, 1))
        self.den.batchScatter(self.pos, vals)
           
        self.den.field /= self.den.volume
        return self.den.field[:, :, :, 0]
