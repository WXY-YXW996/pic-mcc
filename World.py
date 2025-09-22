import numpy as np
import Field

class World(Field.Field):
    def __init__(self, nn):
        super(World,self).__init__(nn)
        
        self.time:float = 0.0
        self.ts:int = 0

    def setTime(self, dt:float, num_ts:int):
        self.dt:float = dt
        self.num_ts:int = num_ts
    
    def advanceTime(self):
        self.time +=self.dt
        self.ts += 1
        return self.ts <= self.num_ts
        
        
