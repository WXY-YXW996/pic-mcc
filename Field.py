import numpy as np
import numba
import h5py


@numba.njit
def gatherField(field_data, loc_pos):
    """
    numba优化的字段插值函数（三线性插值）
    
    参数:
        field_data: np.ndarray - 字段数据 (ni, nj, nk, 3)
        loc_pos: np.ndarray - 局部坐标位置 (3,)
    
    返回:
        np.ndarray - 插值后的字段值 (3,)
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
    
    # 三线性插值
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
def scatterField(field_data, loc_pos, val):
    """
    numba优化的字段散射函数（三线性散射）
    
    参数:
        field_data: np.ndarray - 字段数据 (ni, nj, nk, 3)
        loc_pos: np.ndarray - 局部坐标位置 (3,)
        val: np.ndarray - 要散射的值 (3,)
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
    
    # 三线性散射
    field_data[i,j,k] += val * (1 - di) * (1 - dj) * (1 - dk)
    field_data[i + 1,j,k] += val * (di) * (1 - dj) * (1 - dk)
    field_data[i + 1,j + 1,k] += val * (di) * (dj) * (1 - dk)
    field_data[i,j + 1,k] += val * (1 - di) * (dj) * (1 - dk)
    field_data[i,j,k + 1] += val * (1 - di) * (1 - dj) * (dk)
    field_data[i + 1,j,k + 1] += val * (di) * (1 - dj) * (dk)
    field_data[i + 1,j + 1,k + 1] += val * (di) * (dj) * (dk)
    field_data[i,j + 1,k + 1] += val * (1 - di) * (dj) * (dk)


@numba.njit
def posToLocal(pos, box_min, dh):
    """
    将物理坐标转换为局部网格坐标
    
    参数:
        pos: np.ndarray - 物理位置 (3,)
        box_min: np.ndarray - 网格最小边界 (3,)
        dh: np.ndarray - 网格间距 (3,)
    
    返回:
        np.ndarray - 局部网格坐标 (3,)
    """
    return (pos - box_min) / dh


@numba.njit
def batchGatherField(field_data, positions, box_min, dh):
    """
    批量字段插值函数
    
    参数:
        field_data: np.ndarray - 字段数据 (ni, nj, nk, 3)
        positions: np.ndarray - 粒子位置数组 (N, 3)
        box_min: np.ndarray - 网格最小边界 (3,)
        dh: np.ndarray - 网格间距 (3,)
    
    返回:
        np.ndarray - 插值后的字段值数组 (N, 3)
    """
    n_particles = positions.shape[0]
    result = np.zeros((n_particles, 3))
    
    for i in range(n_particles):
        loc_pos = posToLocal(positions[i], box_min, dh)
        result[i] = gatherField(field_data, loc_pos)
    
    return result


@numba.njit
def batchScatterField(field_data, positions, values, box_min, dh):
    """
    批量字段散射函数
    
    参数:
        field_data: np.ndarray - 字段数据 (ni, nj, nk, 3)
        positions: np.ndarray - 粒子位置数组 (N, 3)
        values: np.ndarray - 要散射的值数组 (N, 3)
        box_min: np.ndarray - 网格最小边界 (3,)
        dh: np.ndarray - 网格间距 (3,)
    """
    n_particles = positions.shape[0]
    
    for i in range(n_particles):
        loc_pos = posToLocal(positions[i], box_min, dh)
        scatterField(field_data, loc_pos, values[i])
        
@numba.njit
def computerVolume(nn, dh):
    """
    计算网格体积
    
    参数:
        nn: tuple - 网格点数 (nx, ny, nz)
        dh: np.ndarray - 网格间距 (3,)
    
    返回:
        float - 网格体积
    """
    
    volume = np.zeros((*nn, 1), dtype=np.float64)
    nx, ny, nz = nn
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                volume[ix,iy,iz] = dh[0] * dh[1] * dh[2]
                if (ix == 0 or ix == nx-1):
                    volume[ix,iy,iz,0] *= 0.5
                if (iy == 0 or iy == ny-1):
                    volume[ix,iy,iz,0] *= 0.5
                if (iz == 0 or iz == nz-1):
                    volume[ix,iy,iz,0] *= 0.5
    return volume



class BaseField:
    def __init__(self, nn):
        """
        初始化BaseField对象
        
        参数:
            nn: int 或 tuple - 网格点数（可以是单个值或三维元组）
        """
        if isinstance(nn, (int, float)):
            self.nn = (int(nn), int(nn), int(nn))
        else:
            self.nn = tuple(nn)
        
        self.box_min = None
        self.box_max = None
        self.dh = None

    def setBox(self, box_min: np.array, box_max: np.array):
        """
        设置box大小
        
        参数:
            box_min: np.array - 下边界 (3,)
            box_max: np.array - 上边界 (3,)
        """
        self.box_min = np.asarray(box_min, dtype=np.float64)
        self.box_max = np.asarray(box_max, dtype=np.float64)
        self.dh = (self.box_max - self.box_min) / (np.array(self.nn) - 1)
        self.volume = computerVolume(self.nn, self.dh)
    
    def XToL(self, pos: np.array):
        """
        将物理坐标转换为局部网格坐标
        
        参数:
            pos: np.array - 物理位置 (3,)
        
        返回:
            np.ndarray - 局部网格坐标 (3,)
        """
        if self.box_min is None or self.dh is None:
            raise ValueError("必须先调用setBox()设置网格边界")
        return posToLocal(pos, self.box_min, self.dh)
    

class Field(BaseField):
    def __init__(self, nn):
        """
        初始化Field对象
        
        参数:
            nn: int 或 tuple - 网格点数
        """
        super(Field, self).__init__(nn)
        self.field = None
        
    def initializeField(self, n_components=3):
        """
        初始化字段数据数组
        
        参数:
            n_components: int - 字段分量数（默认3，对应x,y,z分量）
        """
        self.field = np.zeros((*self.nn, n_components), dtype=np.float64)
        
    def load(self, filename, dataset_name):
        """
        从HDF5文件加载字段数据
        
        参数:
            filename: str - 文件名
            dataset_name: str - 数据集名称
        """
        with h5py.File(filename, "r") as f:
            self.field = f[dataset_name][...]
    
    def save(self, filename, dataset_name):
        """
        保存字段数据到HDF5文件
        
        参数:
            filename: str - 文件名
            dataset_name: str - 数据集名称
        """
        if self.field is None:
            raise ValueError("字段数据未初始化")
        
        with h5py.File(filename, "w") as f:
            f.create_dataset(dataset_name, data=self.field)
    
    def gather(self, loc_pos): 
        """
        在指定位置插值字段值
        
        参数:
            loc_pos: np.ndarray - 局部坐标位置 (3,)
        
        返回:
            np.ndarray - 插值后的字段值
        """
        if self.field is None:
            raise ValueError("字段数据未初始化")
        return gatherField(self.field, loc_pos)
    
    
    def scatter(self, loc_pos, val):
        """
        在指定位置散射值到字段
        
        参数:
            loc_pos: np.ndarray - 局部坐标位置 (3,)
            val: np.ndarray - 要散射的值
        """
        if self.field is None:
            raise ValueError("字段数据未初始化")
        scatterField(self.field, loc_pos, val)
    
    def batchGather(self, positions):
        """
        批量插值字段值
        
        参数:
            positions: np.ndarray - 粒子位置数组 (N, 3)
        
        返回:
            np.ndarray - 插值后的字段值数组 (N, 3)
        """
        if self.field is None:
            raise ValueError("字段数据未初始化")
        if self.box_min is None or self.dh is None:
            raise ValueError("必须先调用setBox()设置网格边界")
        
        return batchGatherField(self.field, positions, self.box_min, self.dh)

    def batchScatter(self, positions, values):
        """
        批量散射值到字段
        
        参数:
            positions: np.ndarray - 粒子位置数组 (N, 3)
            values: np.ndarray - 要散射的值数组 (N, 3)
        """
        if self.field is None:
            raise ValueError("字段数据未初始化")
        if self.box_min is None or self.dh is None:
            raise ValueError("必须先调用setBox()设置网格边界")
            
        batchScatterField(self.field, positions, values, self.box_min, self.dh)
    
    
    def clear(self):
        """
        清零字段数据
        """
        if self.field is not None:
            self.field.fill(0.0)
      
    def getFieldStats(self):
        """
        获取字段统计信息
        
        返回:
            dict - 包含最大值、最小值、平均值等统计信息
        """
        if self.field is None:
            raise ValueError("字段数据未初始化")
        
        return {
            'shape': self.field.shape,
            'max': np.max(self.field),
            'min': np.min(self.field),
            'mean': np.mean(self.field),
            'std': np.std(self.field),
            'norm': np.linalg.norm(self.field)
        }
        
