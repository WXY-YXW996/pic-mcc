import numpy as np
import numba
import h5py


@numba.njit
def gather_field(field_data, loc_pos):
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
def scatter_field(field_data, loc_pos, val):
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
def pos_to_local(pos, box_min, dh):
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
def batch_gather_field(field_data, positions, box_min, dh):
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
        loc_pos = pos_to_local(positions[i], box_min, dh)
        result[i] = gather_field(field_data, loc_pos)
    
    return result


@numba.njit
def batch_scatter_field(field_data, positions, values, box_min, dh):
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
        loc_pos = pos_to_local(positions[i], box_min, dh)
        scatter_field(field_data, loc_pos, values[i])


@numba.njit
def field_gradient(field_data, loc_pos, dh):
    """
    计算字段梯度（数值微分）
    
    参数:
        field_data: np.ndarray - 字段数据 (ni, nj, nk, 3)
        loc_pos: np.ndarray - 局部坐标位置 (3,)
        dh: np.ndarray - 网格间距 (3,)
    
    返回:
        np.ndarray - 梯度张量 (3, 3) [grad_x, grad_y, grad_z]
    """
    gradient = np.zeros((3, 3))
    eps = 0.5  # 有限差分步长
    
    for dim in range(3):
        # 前向差分
        pos_forward = loc_pos.copy()
        pos_forward[dim] += eps
        field_forward = gather_field(field_data, pos_forward)
        
        # 后向差分
        pos_backward = loc_pos.copy()
        pos_backward[dim] -= eps
        field_backward = gather_field(field_data, pos_backward)
        
        # 中心差分
        gradient[:, dim] = (field_forward - field_backward) / (2.0 * eps * dh[dim])
    
    return gradient


@numba.njit
def field_divergence(field_data, loc_pos, dh):
    """
    计算字段散度
    
    参数:
        field_data: np.ndarray - 字段数据 (ni, nj, nk, 3)
        loc_pos: np.ndarray - 局部坐标位置 (3,)
        dh: np.ndarray - 网格间距 (3,)
    
    返回:
        float - 散度值
    """
    gradient = field_gradient(field_data, loc_pos, dh)
    return gradient[0, 0] + gradient[1, 1] + gradient[2, 2]


@numba.njit
def field_curl(field_data, loc_pos, dh):
    """
    计算字段旋度
    
    参数:
        field_data: np.ndarray - 字段数据 (ni, nj, nk, 3)
        loc_pos: np.ndarray - 局部坐标位置 (3,)
        dh: np.ndarray - 网格间距 (3,)
    
    返回:
        np.ndarray - 旋度向量 (3,)
    """
    gradient = field_gradient(field_data, loc_pos, dh)
    curl = np.zeros(3)
    
    # curl = ∇ × F
    curl[0] = gradient[2, 1] - gradient[1, 2]  # ∂Fz/∂y - ∂Fy/∂z
    curl[1] = gradient[0, 2] - gradient[2, 0]  # ∂Fx/∂z - ∂Fz/∂x
    curl[2] = gradient[1, 0] - gradient[0, 1]  # ∂Fy/∂x - ∂Fx/∂y
    
    return curl


@numba.njit
def smooth_field(field_data, iterations=1):
    """
    字段平滑处理（简单的邻域平均）
    
    参数:
        field_data: np.ndarray - 字段数据 (ni, nj, nk, 3)
        iterations: int - 平滑迭代次数
    """
    ni, nj, nk, ncomp = field_data.shape
    temp_field = np.zeros_like(field_data)
    
    for _ in range(iterations):
        for i in range(1, ni-1):
            for j in range(1, nj-1):
                for k in range(1, nk-1):
                    for c in range(ncomp):
                        # 27点邻域平均
                        sum_val = 0.0
                        count = 0
                        for di in range(-1, 2):
                            for dj in range(-1, 2):
                                for dk in range(-1, 2):
                                    sum_val += field_data[i+di, j+dj, k+dk, c]
                                    count += 1
                        temp_field[i, j, k, c] = sum_val / count
        
        # 复制回原数组
        field_data[:] = temp_field[:]


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
        return pos_to_local(pos, self.box_min, self.dh)


class Field(BaseField):
    def __init__(self, nn):
        """
        初始化Field对象
        
        参数:
            nn: int 或 tuple - 网格点数
        """
        super(Field, self).__init__(nn)
        self.field = None
        
    def initialize_field(self, n_components=3):
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
        return gather_field(self.field, loc_pos)
    
    def scatter(self, loc_pos, val):
        """
        在指定位置散射值到字段
        
        参数:
            loc_pos: np.ndarray - 局部坐标位置 (3,)
            val: np.ndarray - 要散射的值
        """
        if self.field is None:
            raise ValueError("字段数据未初始化")
        scatter_field(self.field, loc_pos, val)
    
    def batch_gather(self, positions):
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
        
        return batch_gather_field(self.field, positions, self.box_min, self.dh)
    
    def batch_scatter(self, positions, values):
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
        
        batch_scatter_field(self.field, positions, values, self.box_min, self.dh)
    
    def get_gradient(self, pos):
        """
        计算指定位置的字段梯度
        
        参数:
            pos: np.ndarray - 物理位置 (3,)
        
        返回:
            np.ndarray - 梯度张量 (3, 3)
        """
        if self.field is None:
            raise ValueError("字段数据未初始化")
        
        loc_pos = self.XToL(pos)
        return field_gradient(self.field, loc_pos, self.dh)
    
    def get_divergence(self, pos):
        """
        计算指定位置的字段散度
        
        参数:
            pos: np.ndarray - 物理位置 (3,)
        
        返回:
            float - 散度值
        """
        if self.field is None:
            raise ValueError("字段数据未初始化")
        
        loc_pos = self.XToL(pos)
        return field_divergence(self.field, loc_pos, self.dh)
    
    def get_curl(self, pos):
        """
        计算指定位置的字段旋度
        
        参数:
            pos: np.ndarray - 物理位置 (3,)
        
        返回:
            np.ndarray - 旋度向量 (3,)
        """
        if self.field is None:
            raise ValueError("字段数据未初始化")
        
        loc_pos = self.XToL(pos)
        return field_curl(self.field, loc_pos, self.dh)
    
    def smooth(self, iterations=1):
        """
        对字段进行平滑处理
        
        参数:
            iterations: int - 平滑迭代次数
        """
        if self.field is None:
            raise ValueError("字段数据未初始化")
        
        smooth_field(self.field, iterations)
    
    def clear(self):
        """
        清零字段数据
        """
        if self.field is not None:
            self.field.fill(0.0)
    
    def get_field_stats(self):
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
