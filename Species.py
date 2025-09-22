import numpy as np
import World
import Particles



class Species:
    def __init__(self, speics_name:str, mass:float, charge:float, world:World.World):
        self.name = speics_name
        self.mass = mass
        self.charge = charge
        self.world = world
        self.particles = np.empty((1),dtype= Particles.Particles)
        