"""
PIC-MCC 仿真输出模块
支持VTK格式输出，用于ParaView等后处理软件
包括粒子数据、场数据的输出和诊断数据的保存
"""

import numpy as np
import h5py
import os
import time
from mpi4py import MPI
import xml.etree.ElementTree as ET
import xml.dom.minidom as minidom

try:
    import vtk
    from vtkmodules.util.numpy_support import numpy_to_vtk
    VTK_AVAILABLE = True
except ImportError:
    VTK_AVAILABLE = False
    print("警告: 未安装VTK库，将使用简化的XML格式输出")


class VTKOutput:
    """
    VTK格式输出类
    支持结构化网格(vts)和无结构化网格(vtu)输出
    """
    
    def __init__(self, output_dir="vtk_output", base_name="simulation"):
        """
        初始化VTK输出器
        
        参数:
            output_dir: str - 输出目录
            base_name: str - 文件基础名称
        """
        self.output_dir = output_dir
        self.base_name = base_name
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        
        # 只有根进程创建输出目录
        if self.rank == 0:
            os.makedirs(output_dir, exist_ok=True)
            os.makedirs(f"{output_dir}/fields", exist_ok=True)
            os.makedirs(f"{output_dir}/particles", exist_ok=True)
            os.makedirs(f"{output_dir}/diagnostics", exist_ok=True)
        
        # 等待目录创建完成
        self.comm.Barrier()
    
    def write_structured_grid(self, field, timestep, field_name="field", 
                            field_type="vector"):
        """
        写入结构化网格数据（VTS格式）
        
        参数:
            field: Field对象 - 要输出的场
            timestep: int - 时间步
            field_name: str - 场名称
            field_type: str - 场类型 ("vector" 或 "scalar")
        """
        filename = f"{self.output_dir}/fields/{field_name}_t{timestep:06d}.vts"
        
        if self.rank == 0:
            self._write_vts_vtk(field, filename, field_name, field_type)

    
    def write_unstructured_grid(self, species_list, timestep):
        """
        写入非结构化网格数据（VTU格式）- 粒子数据
        
        参数:
            species_list: list - 粒子种类列表
            timestep: int - 时间步
        """
        filename = f"{self.output_dir}/particles/particles_t{timestep:06d}.vtu"

        if self.rank == 0:
            self._write_vtu_vtk(species_list, filename)
        
    
    def _write_vts_vtk(self, field, filename, field_name, field_type):
        """
        使用VTK库写入结构化网格
        """
        if field.field is None:
            return
        
        # 创建结构化网格
        grid = vtk.vtkStructuredGrid()
        
        # 设置网格尺寸
        nx, ny, nz = field.nn
        grid.SetDimensions(nx, ny, nz)
        
        # 创建点坐标
        points = vtk.vtkPoints()
        for k in range(nz):
            for j in range(ny):
                for i in range(nx):
                    x = field.box_min[0] + i * field.dh[0]
                    y = field.box_min[1] + j * field.dh[1]
                    z = field.box_min[2] + k * field.dh[2]
                    points.InsertNextPoint(x, y, z)
        
        grid.SetPoints(points)
        
        # 添加场数据
        if field_type == "vector" and len(field.field.shape) == 4:
            # 矢量场
            field_data = field.field.flatten(order='F').reshape(-1, 3)
            vtk_array = numpy_to_vtk(field_data)
            vtk_array.SetName(field_name)
            grid.GetPointData().SetVectors(vtk_array)
        else:
            # 标量场
            if len(field.field.shape) == 4:
                field_data = np.linalg.norm(field.field, axis=3).flatten(order='F')
            else:
                field_data = field.field.flatten(order='F')
            vtk_array = numpy_to_vtk(field_data)
            vtk_array.SetName(field_name)
            grid.GetPointData().SetScalars(vtk_array)
        
        # 写入文件
        writer = vtk.vtkXMLStructuredGridWriter()
        writer.SetFileName(filename)
        writer.SetInputData(grid)
        writer.Write()
    
    def _write_vts_xml(self, field, filename, field_name, field_type):
        """
        手动写入VTS XML格式
        """
        if field.field is None:
            return
        
        nx, ny, nz = field.nn
        
        # 创建XML结构
        root = ET.Element("VTKFile", type="StructuredGrid", version="0.1", 
                         byte_order="LittleEndian")
        structured_grid = ET.SubElement(root, "StructuredGrid", 
                                      WholeExtent=f"0 {nx-1} 0 {ny-1} 0 {nz-1}")
        piece = ET.SubElement(structured_grid, "Piece", 
                            Extent=f"0 {nx-1} 0 {ny-1} 0 {nz-1}")
        
        # 添加点坐标
        points_elem = ET.SubElement(piece, "Points")
        points_data = ET.SubElement(points_elem, "DataArray", type="Float32", 
                                  NumberOfComponents="3", format="ascii")
        
        points_text = ""
        for k in range(nz):
            for j in range(ny):
                for i in range(nx):
                    x = field.box_min[0] + i * field.dh[0]
                    y = field.box_min[1] + j * field.dh[1]
                    z = field.box_min[2] + k * field.dh[2]
                    points_text += f"{x} {y} {z}\n"
        points_data.text = points_text
        
        # 添加场数据
        point_data = ET.SubElement(piece, "PointData")
        
        if field_type == "vector" and len(field.field.shape) == 4:
            field_data_elem = ET.SubElement(point_data, "DataArray", 
                                          type="Float32", Name=field_name, 
                                          NumberOfComponents="3", format="ascii")
            field_text = ""
            for k in range(nz):
                for j in range(ny):
                    for i in range(nx):
                        field_text += f"{field.field[i,j,k,0]} {field.field[i,j,k,1]} {field.field[i,j,k,2]}\n"
        else:
            field_data_elem = ET.SubElement(point_data, "DataArray", 
                                          type="Float32", Name=field_name, 
                                          format="ascii")
            field_text = ""
            if len(field.field.shape) == 4:
                field_mag = np.linalg.norm(field.field, axis=3)
            else:
                field_mag = field.field
            
            for k in range(nz):
                for j in range(ny):
                    for i in range(nx):
                        field_text += f"{field_mag[i,j,k]}\n"
        
        field_data_elem.text = field_text
        
        # 写入文件
        rough_string = ET.tostring(root, 'unicode')
        reparsed = minidom.parseString(rough_string)
        with open(filename, 'w') as f:
            f.write(reparsed.toprettyxml(indent="  "))
    
    def _write_vtu_vtk(self, species_list, filename):
        """
        使用VTK库写入非结构化网格（粒子数据）
        """
        # 收集所有粒子数据
        all_positions = []
        all_velocities = []
        all_kinetic = []
        all_species_ids = []
        all_masses = []
        all_charges = []
        
        for species_id, species in enumerate(species_list):
            if species.npar > 0:
                all_positions.append(species.pos)
                all_velocities.append(species.vel)
                all_kinetic.append(species.kinetic)
                all_species_ids.extend([species_id] * species.npar)
                all_masses.extend([species.mass] * species.npar)
                all_charges.extend([species.charge] * species.npar)
        
        if not all_positions:
            return
        
        positions = np.vstack(all_positions)
        velocities = np.vstack(all_velocities)
        kinetic = np.vstack(all_kinetic)
        total_particles = len(positions)
        
        # 创建非结构化网格
        grid = vtk.vtkUnstructuredGrid()
        
        # 添加点
        points = vtk.vtkPoints()
        for pos in positions:
            points.InsertNextPoint(pos[0], pos[1], pos[2])
        grid.SetPoints(points)
        
        # 添加顶点单元（每个粒子一个顶点）
        for i in range(total_particles):
            vertex = vtk.vtkVertex()
            vertex.GetPointIds().SetId(0, i)
            grid.InsertNextCell(vertex.GetCellType(), vertex.GetPointIds())
        
        # 添加数据数组
        velocity_array = numpy_to_vtk(velocities)
        velocity_array.SetName("Velocity")
        grid.GetPointData().SetVectors(velocity_array)
        
        species_array = numpy_to_vtk(np.array(all_species_ids))
        species_array.SetName("Species_ID")
        grid.GetPointData().AddArray(species_array)
        
        mass_array = numpy_to_vtk(np.array(all_masses))
        mass_array.SetName("Mass")
        grid.GetPointData().AddArray(mass_array)
        
        charge_array = numpy_to_vtk(np.array(all_charges))
        charge_array.SetName("Charge")
        grid.GetPointData().AddArray(charge_array)
        
        # 计算动能
        kinetic_array = numpy_to_vtk(kinetic)
        kinetic_array.SetName("Kinetic_Energy")
        grid.GetPointData().AddArray(kinetic_array)
        
        # 写入文件
        writer = vtk.vtkXMLUnstructuredGridWriter()
        writer.SetFileName(filename)
        writer.SetInputData(grid)
        writer.Write()
    
    def _write_vtu_xml(self, species_list, filename):
        """
        手动写入VTU XML格式
        """
        # 收集所有粒子数据
        all_positions = []
        all_velocities = []
        all_species_ids = []
        all_masses = []
        all_charges = []
        
        for species_id, species in enumerate(species_list):
            if species.npar > 0:
                all_positions.append(species.pos)
                all_velocities.append(species.vel)
                all_species_ids.extend([species_id] * species.npar)
                all_masses.extend([species.mass] * species.npar)
                all_charges.extend([species.charge] * species.npar)
        
        if not all_positions:
            return
        
        positions = np.vstack(all_positions)
        velocities = np.vstack(all_velocities)
        total_particles = len(positions)
        
        # 创建XML结构
        root = ET.Element("VTKFile", type="UnstructuredGrid", version="0.1", 
                         byte_order="LittleEndian")
        unstructured_grid = ET.SubElement(root, "UnstructuredGrid")
        piece = ET.SubElement(unstructured_grid, "Piece", 
                            NumberOfPoints=str(total_particles), 
                            NumberOfCells=str(total_particles))
        
        # 添加点坐标
        points_elem = ET.SubElement(piece, "Points")
        points_data = ET.SubElement(points_elem, "DataArray", type="Float32", 
                                  NumberOfComponents="3", format="ascii")
        points_text = ""
        for pos in positions:
            points_text += f"{pos[0]} {pos[1]} {pos[2]}\n"
        points_data.text = points_text
        
        # 添加单元定义（顶点单元）
        cells_elem = ET.SubElement(piece, "Cells")
        
        # 连接性
        connectivity = ET.SubElement(cells_elem, "DataArray", type="Int32", 
                                   Name="connectivity", format="ascii")
        conn_text = " ".join([str(i) for i in range(total_particles)])
        connectivity.text = conn_text
        
        # 偏移量
        offsets = ET.SubElement(cells_elem, "DataArray", type="Int32", 
                              Name="offsets", format="ascii")
        offset_text = " ".join([str(i+1) for i in range(total_particles)])
        offsets.text = offset_text
        
        # 单元类型（1 = VTK_VERTEX）
        types = ET.SubElement(cells_elem, "DataArray", type="UInt8", 
                            Name="types", format="ascii")
        types_text = " ".join(["1"] * total_particles)
        types.text = types_text
        
        # 添加点数据
        point_data = ET.SubElement(piece, "PointData")
        
        # 速度
        velocity_data = ET.SubElement(point_data, "DataArray", type="Float32", 
                                    Name="Velocity", NumberOfComponents="3", 
                                    format="ascii")
        vel_text = ""
        for vel in velocities:
            vel_text += f"{vel[0]} {vel[1]} {vel[2]}\n"
        velocity_data.text = vel_text
        
        # 物种ID
        species_data = ET.SubElement(point_data, "DataArray", type="Int32", 
                                   Name="Species_ID", format="ascii")
        species_text = " ".join([str(sid) for sid in all_species_ids])
        species_data.text = species_text
        
        # 质量
        mass_data = ET.SubElement(point_data, "DataArray", type="Float32", 
                                Name="Mass", format="ascii")
        mass_text = " ".join([str(m) for m in all_masses])
        mass_data.text = mass_text
        
        # 电荷
        charge_data = ET.SubElement(point_data, "DataArray", type="Float32", 
                                  Name="Charge", format="ascii")
        charge_text = " ".join([str(c) for c in all_charges])
        charge_data.text = charge_text
        
        # 动能
        vel_mag = np.linalg.norm(velocities, axis=1)
        kinetic_energy = 0.5 * np.array(all_masses) * vel_mag**2
        kinetic_data = ET.SubElement(point_data, "DataArray", type="Float32", 
                                   Name="Kinetic_Energy", format="ascii")
        kinetic_text = " ".join([str(ke) for ke in kinetic_energy])
        kinetic_data.text = kinetic_text
        
        # 写入文件
        rough_string = ET.tostring(root, 'unicode')
        reparsed = minidom.parseString(rough_string)
        with open(filename, 'w') as f:
            f.write(reparsed.toprettyxml(indent="  "))
    
    def write_parallel_collection(self, timestep, data_type="particles"):
        """
        写入并行集合文件（PVTU/PVTS）
        
        参数:
            timestep: int - 时间步
            data_type: str - 数据类型 ("particles" 或 "fields")
        """
        if self.rank != 0:
            return
        
        if data_type == "particles":
            self._write_pvtu_collection(timestep)
        elif data_type == "fields":
            self._write_pvtu_collection(timestep)
    
    def _write_pvtu_collection(self, timestep):
        """
        写入PVTU集合文件
        """
        filename = f"{self.output_dir}/particles/particles_t{timestep:06d}.pvtu"
        
        root = ET.Element("VTKFile", type="PUnstructuredGrid", version="0.1")
        punstructured_grid = ET.SubElement(root, "PUnstructuredGrid", GhostLevel="0")
        
        # 添加点数据描述
        ppoint_data = ET.SubElement(punstructured_grid, "PPointData")
        ET.SubElement(ppoint_data, "PDataArray", type="Float32", Name="Velocity", 
                     NumberOfComponents="3")
        ET.SubElement(ppoint_data, "PDataArray", type="Int32", Name="Species_ID")
        ET.SubElement(ppoint_data, "PDataArray", type="Float32", Name="Mass")
        ET.SubElement(ppoint_data, "PDataArray", type="Float32", Name="Charge")
        ET.SubElement(ppoint_data, "PDataArray", type="Float32", Name="Kinetic_Energy")
        
        # 添加点描述
        ppoints = ET.SubElement(punstructured_grid, "PPoints")
        ET.SubElement(ppoints, "PDataArray", type="Float32", NumberOfComponents="3")
        
        # 添加各个进程的文件
        for rank in range(self.size):
            piece_filename = f"particles_t{timestep:06d}_r{rank:04d}.vtu"
            ET.SubElement(punstructured_grid, "Piece", Source=piece_filename)
        
        # 写入文件
        rough_string = ET.tostring(root, 'unicode')
        reparsed = minidom.parseString(rough_string)
        with open(filename, 'w') as f:
            f.write(reparsed.toprettyxml(indent="  "))


class DiagnosticOutput:
    """
    诊断数据输出类
    """
    
    def __init__(self, output_dir="diagnostics", base_name="diagnostics"):
        """
        初始化诊断输出器
        
        参数:
            output_dir: str - 输出目录
            base_name: str - 文件基础名称
        """
        self.output_dir = output_dir
        self.base_name = base_name
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        
        if self.rank == 0:
            os.makedirs(output_dir, exist_ok=True)
        
        self.comm.Barrier()
        
        # 初始化历史数据存储
        self.time_history = []
        self.energy_history = []
        self.particle_count_history = []
        self.field_energy_history = []
    
    def collect_timestep_data(self, world, species_list, fields=None):
        """
        收集时间步诊断数据
        
        参数:
            world: World对象
            species_list: list - 粒子种类列表
            fields: dict - 场字典
        """
        # 收集基本信息
        self.time_history.append(world.time)
        
        # 收集粒子信息
        total_particles = sum([species.npar for species in species_list])
        particle_counts = [species.npar for species in species_list]
        self.particle_count_history.append(particle_counts)
        
        # 计算总动能
        total_kinetic = 0.0
        species_energies = []
        for species in species_list:
            if species.npar > 0:
                vel_mag = np.linalg.norm(species.vel, axis=1)
                kinetic = 0.5 * species.mass * np.sum(vel_mag**2)
                species_energies.append(kinetic)
                total_kinetic += kinetic
            else:
                species_energies.append(0.0)
        
        self.energy_history.append(species_energies)
        
        # 收集场能量（如果提供）
        field_energies = {}
        if fields:
            for field_name, field_obj in fields.items():
                if field_obj.field is not None:
                    field_energy = 0.5 * np.sum(field_obj.field**2)
                    field_energies[field_name] = field_energy
        
        self.field_energy_history.append(field_energies)
    
    def save_time_series(self, filename=None):
        """
        保存时间序列数据
        
        参数:
            filename: str - 文件名（可选）
        """
        if self.rank != 0:
            return
        
        if filename is None:
            filename = f"{self.output_dir}/{self.base_name}_timeseries.h5"
        
        with h5py.File(filename, 'w') as f:
            # 保存时间数据
            f.create_dataset('time', data=np.array(self.time_history))
            
            # 保存粒子数量历史
            if self.particle_count_history:
                particle_counts = np.array(self.particle_count_history)
                f.create_dataset('particle_counts', data=particle_counts)
            
            # 保存能量历史
            if self.energy_history:
                energies = np.array(self.energy_history)
                f.create_dataset('kinetic_energies', data=energies)
            
            # 保存场能量历史
            if self.field_energy_history:
                for field_name in self.field_energy_history[0].keys():
                    field_energy_data = [fe.get(field_name, 0.0) for fe in self.field_energy_history]
                    f.create_dataset(f'field_energy_{field_name}', data=np.array(field_energy_data))
            
            # 保存元数据
            f.attrs['description'] = 'PIC-MCC simulation time series data'
            f.attrs['total_timesteps'] = len(self.time_history)
            f.attrs['final_time'] = self.time_history[-1] if self.time_history else 0.0
        
        print(f"时间序列数据已保存到: {filename}")
    
    def save_final_state(self, world, species_list, fields=None, filename=None):
        """
        保存最终状态数据
        
        参数:
            world: World对象
            species_list: list - 粒子种类列表
            fields: dict - 场字典
            filename: str - 文件名（可选）
        """
        if self.rank != 0:
            return
        
        if filename is None:
            filename = f"{self.output_dir}/{self.base_name}_final_state.h5"
        
        with h5py.File(filename, 'w') as f:
            # 保存仿真参数
            sim_group = f.create_group('simulation')
            sim_group.create_dataset('final_time', data=world.time)
            sim_group.create_dataset('final_timestep', data=world.ts)
            sim_group.create_dataset('dt', data=world.dt)
            sim_group.create_dataset('box_min', data=world.box_min)
            sim_group.create_dataset('box_max', data=world.box_max)
            
            # 保存粒子数据
            particles_group = f.create_group('particles')
            for i, species in enumerate(species_list):
                species_group = particles_group.create_group(f'species_{i}')
                species_group.attrs['name'] = species.name
                species_group.attrs['mass'] = species.mass
                species_group.attrs['charge'] = species.charge
                species_group.attrs['npar'] = species.npar
                
                if species.npar > 0:
                    species_group.create_dataset('positions', data=species.pos)
                    species_group.create_dataset('velocities', data=species.vel)
            
            # 保存场数据
            if fields:
                fields_group = f.create_group('fields')
                for field_name, field_obj in fields.items():
                    if field_obj.field is not None:
                        field_group = fields_group.create_group(field_name)
                        field_group.create_dataset('data', data=field_obj.field)
                        field_group.create_dataset('box_min', data=field_obj.box_min)
                        field_group.create_dataset('box_max', data=field_obj.box_max)
                        field_group.create_dataset('grid_size', data=field_obj.nn)
        
        print(f"最终状态数据已保存到: {filename}")


def create_paraview_state_file(output_dir, timesteps, base_name="simulation"):
    """
    创建ParaView状态文件，便于批量加载数据
    
    参数:
        output_dir: str - 输出目录
        timesteps: list - 时间步列表
        base_name: str - 文件基础名称
    """
    state_filename = f"{output_dir}/{base_name}_paraview_state.py"
    
    with open(state_filename, 'w') as f:
        f.write("""
# ParaView state file for PIC-MCC simulation
# Load this script in ParaView to quickly set up visualization

from paraview.simple import *

# 清除现有数据
Delete(GetSources())

# 加载粒子数据
particle_files = []
""")
        
        for ts in timesteps:
            f.write(f'particle_files.append("{output_dir}/particles/particles_t{ts:06d}.pvtu")\n')
        
        f.write("""
# 创建粒子数据读取器
particle_reader = XMLPartitionedUnstructuredGridReader(FileName=particle_files)
particle_reader.UpdatePipeline()

# 创建粒子显示
particle_display = Show(particle_reader)
particle_display.Representation = 'Points'
particle_display.PointSize = 2.0

# 按物种ID着色
ColorBy(particle_display, ('POINTS', 'Species_ID'))

# 加载场数据（如果存在）
field_files = []
""")
        
        for ts in timesteps:
            f.write(f'field_files.append("{output_dir}/fields/magnetic_field_t{ts:06d}.pvts")\n')
        
        f.write("""
if field_files:
    field_reader = XMLPartitionedStructuredGridReader(FileName=field_files)
    field_reader.UpdatePipeline()
    
    # 创建场显示
    field_display = Show(field_reader)
    field_display.Representation = 'Outline'

# 设置视图
renderView = GetActiveViewOrCreate('RenderView')
renderView.ResetCamera()

print("ParaView state loaded successfully!")
""")
    
    print(f"ParaView状态文件已创建: {state_filename}")
