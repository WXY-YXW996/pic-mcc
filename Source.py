import numpy as np
import Species
import World


class Source:
    def __init__(self, sp:Species.Species, world:World.World):
        self.sp = sp
        self.world = world
        
    def sample(self, npar):
        self.sp.npar = npar
            
    