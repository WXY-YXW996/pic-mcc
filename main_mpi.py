from mpi4py import MPI
import numpy as np
import h5py
import pandas as pd
from scipy.interpolate import griddata
import numba
import matplotlib.pyplot as plt
import time

import World
import Field
import Species
import Source
import Parameters
import config  # 导入配置文件

# 从配置文件导入参数
nn = config.nn
nn_b = config.nn_b
nn_e = config.nn_e
box_min, box_max = config.box_min, config.box_max
dt, Nt = config.dt, config.Nt
npar = config.npar

##########################

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

start_time = time.time()

world = World.World(nn)
world.setBox(box_min, box_max)
world.setTime(dt, Nt)


b = Field.Field(nn_b)
b.setBox(world.box_min,world.box_max)
b.load(config.data_file, config.magnetic_field_dataset)

e = Field.Field(nn_e)
e.setBox(world.box_min,world.box_max)
# e.load(config.data_file, config.electric_field_dataset)

species = [Species.Species("e-", Parameters.ME, -1. * Parameters.QE, b, e, world)]
        #    Species.Species("Ar", 40 * Parameters.MAU, 0.0 * Parameters.QE, b, e, world),
        #    Species.Species("Ar+", 40 * Parameters.MAU, 1.0 * Parameters.QE, b, e, world)]

species[0].addParticles(np.array([-0.0625,0.0,0.0]),np.array([0.0,0.0,100]))


electron_history=[]

while(world.advanceTime()):
    for sp in species:
        # electron_history.append(sp.pos.copy())
        sp.advance()
        sp.boundary()

if (rank == 0):
    print("message from rank ",rank," running time: ",time.time() - start_time, " seconds")


source = [Source.Source(species[0], world)]






MPI.Finalize()
