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
import OutputModule
import config  # 导入配置文件


# 从配置文件导入参数
nn = config.nn
nn_b = config.nn_b
nn_e = config.nn_e
box_min, box_max = config.box_min, config.box_max
dt, Nt = config.dt, config.Nt
npar_e = config.npar_e
weight_e = config.weight_e
fre = config.fre

##########################

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

start_time = time.time()

world = World.World(nn, box_min, box_max)
world.setTime(dt, Nt)
world.setFre(fre)


b = Field.Field(nn_b)
b.setBox(world.box_min, world.box_max)
b.load(config.data_file, config.magnetic_field_dataset)

e_r = Field.Field(nn_e)
e_r.setBox(world.box_min, world.box_max)
e_r.load(config.data_file, config.electric_field_real_dataset)

e_i = Field.Field(nn_e)
e_i.setBox(world.box_min, world.box_max)
e_i.load(config.data_file, config.electric_field_imag_dataset)

species = [
    Species.Species(
        "e-", Parameters.ME, -1.0 * Parameters.QE, weight_e, b, e_r, e_i, world
    )
]
#    Species.Species("Ar", 40 * Parameters.MAU, 0.0 * Parameters.QE, b, e, world),
#    Species.Species("Ar+", 40 * Parameters.MAU, 1.0 * Parameters.QE, b, e, world)]


source = [Source.Source(sp, world) for sp in species]

source[0].sampleInECR(npar_e, b)


print("initialization done")

# 可选：初始化简单输出
if hasattr(config, "save_vtk") and config.save_vtk:
    vtk_output = OutputModule.VTKOutput("vtk_output", "pic_mcc_simple")

for sp in species:
    sp.computeNumberDensity()
world.computeRho(species)

vtk_output.write_unstructured_grid(species, world.ts)
vtk_output.write_structured_grid(world.rho, world.ts, "charge_density", "scalar")
# vtk_output.write_structured_grid(species[0].den, world.ts, "num_density", "scalar")

while world.advanceTime():
    for sp in species:
        sp.advance()
        sp.boundary()
        sp.computeNumberDensity()

    # 定期计算电荷密度，求解possion方程
    if world.ts % config.output_interval == 0:
        world.computeRho(species)
        world.potentialSlover()

    # 可选：定期输出VTK文件
    if (
        hasattr(config, "save_vtk")
        and config.save_vtk
        and hasattr(config, "output_interval")
        and world.ts % config.output_interval == 0
        and rank == 0
    ):
        print(f"rank {rank} output vtk file at time {world.ts}")
        vtk_output.write_unstructured_grid(species, world.ts)
        vtk_output.write_structured_grid(
            world.rho, world.ts, "charge_density", "scalar"
        )


if rank == 0:

    print(
        "message from rank ",
        rank,
        " running time: ",
        time.time() - start_time,
        " seconds",
    )


