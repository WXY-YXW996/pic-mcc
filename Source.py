import numpy as np
import Species
import World
import Field


class Source:
    def __init__(self, sp:Species.Species, world:World.World):
        self.sp = sp
        self.world = world
        
    def sample(self, npar):
        for i in range(npar):
            pos = np.random.uniform(self.world.box_min, self.world.box_max)
            vel = np.random.normal(1e5, 1e6, size=3)  # 这里假设速度服从均值为0，标准差为1e5的正态分布
            self.sp.addParticles(pos, vel)
    
    def sampleInECR(self, npar, b:Field.Field):
        for i in range(npar):
            pos = self.inECR(b)
            vel = np.random.normal(1e5, 1e6, size=3)
            self.sp.addParticles(pos, vel)
            
    
    def inECR(self, b:Field.Field):
        inecr = False
        while(not inecr):
            pos = np.random.uniform(self.world.box_min, self.world.box_max)
            loc = b.XToL(pos)
            b_part = b.gather(loc)
            if np.linalg.norm(b_part) < self.world.b_ecr:
                inecr = True
        return pos