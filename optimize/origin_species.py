import numpy as np
import numba

import World
import Field
import scipy.constants as constant


class Particles:
    def __init__(self, pos: np.ndarray, vel: np.ndarray) -> None:
        """
        初始化粒子对象
        
        参数:
            pos: np.ndarray - 粒子位置坐标数组
            vel: np.ndarray - 粒子速度向量数组
        """
        self.pos = pos
        self.vel = vel



class Species:
    def __init__(self, species_name:str, mass:float, charge:float, b:Field.Field, e:Field.Field, world:World.World):
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
        self.b = b
        self.e = e
        self.world = world
        self.particles = np.empty((0),dtype = Particles)
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
        new_particle = Particles(pos, vel)
        self.particles = np.append(self.particles, np.array([new_particle], dtype=Particles))
    

    def advance(self) -> None:
        """
        Boris推进器推进所有粒子
        参考资料: Birdsall, C. K., & Langdon, A. B. (2004). Plasma physics via computer simulation. CRC Press.
        """
        dt = self.world.dt
        # box_min = self.world.box_min
        # box_max = self.world.box_max
        
        for part in self.particles:
            lc_e = self.e.XToL(part.pos)
            e_part = np.array([0.0,0.0,0.0]) 
            # e_part = self.e.gather(lc_e)
            
            lc_b = self.b.XToL(part.pos)
            b_part = self.b.gather(lc_b)
            
            ## boris mover            
            ff = self.charge / self.mass * dt / 2
            
            v_minus = part.vel + ff * e_part
            
            t = ff * b_part
            
            v_prime = v_minus + np.cross(v_minus, t)
            
            s = 2.0 * t / (1 + np.dot(t,t))
            
            v_plus = v_minus + np.cross(v_prime, s)
            
            part.vel = v_plus + ff * e_part
            
            part.pos += part.vel * dt
            
    def boundary(self) -> None:
        """
        处理粒子边界条件，简单反弹边界
        """
        for part in self.particles:
            for i in range(3):
                if part.pos[i] < self.world.box_min[i]:
                    part.pos[i] = self.world.box_min[i]
                    part.vel[i] = -part.vel[i]
                elif part.pos[i] > self.world.box_max[i]:
                    part.pos[i] = self.world.box_max[i]
                    part.vel[i] = -part.vel[i]
    
    def getKinetic(self):
        """
        计算所有粒子的动能
        
        返回: np.ndarray - 粒子动能数组
        """
        kinetic = np.zeros(self.particles.size)
        
        for i in range(self.particles.size):
            beta = self.particles[i].vel / constant.c
            gamma = 1./np.sqrt(1 + np.dot(beta, beta))
            kinetic[i] = self.mass * constant.c**2 * (gamma - 1)
            
        self.kinetic = kinetic
        return kinetic
    
    def getMomentum(self):
        """
        计算所有粒子的动量
        
        返回: np.ndarray - 粒子动量数组
        """
        momentum = np.zeros((self.particles.size, 3))
        
        for i in range(self.particles.size):
            beta = self.particles[i].vel / constant.c
            gamma = 1./np.sqrt(1 + np.dot(beta, beta))
            momentum[i] = self.mass * gamma * self.particles[i].vel
            
        self.momentum = momentum
        return momentum