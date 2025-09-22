import numpy as np
import numba
import h5py

@numba.njit
def gather_field(field_data, loc_pos):
    """
    numba优化的字段插值函数
    
    参数:
        field_data: np.ndarray - 字段数据
        loc_pos: np.ndarray - 局部坐标位置
    
    返回:
        np.ndarray - 插值后的字段值
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
def pos_to_local(pos, box_min, dh):
    """
    将物理坐标转换为局部网格坐标
    
    参数:
        pos: np.ndarray - 物理位置
        box_min: np.ndarray - 网格最小边界
        dh: np.ndarray - 网格间距
    
    返回:
        np.ndarray - 局部网格坐标
    """
    return (pos - box_min) / dh

class BaseField:
    def __init__(self, nn):
        """
        初始化BaseField对象
        
        参数:
            nn: int - 网格点数
        """
        self.nn = nn

    def setBox(self, box_min:np.array, box_max:np.array):
        """
        设置box大小
        
        参数:
            box_min: np.array - 下边界
            box_max: np.array - 上边界
        """
        self.box_min = box_min
        self.box_max = box_max
        self.dh = (box_max - box_min) / (self.nn - 1)
    
    def XToL(self, pos:np.array):
        lc = np.zeros(3)
        lc = pos_to_local(pos, self.box_min, self.dh)
        return lc


class Field(BaseField):
    def __init__(self, nn):
        super(Field,self).__init__(nn)
        
    def load(self, filename, dataset_name):
        with h5py.File(filename, "r") as f:
            self.field = f[dataset_name][...]
    
    def gather(self, loc_pos): 
        val = gather_field(self.field, loc_pos)
        return val
    
    def scatter(self,loc_pos,val):
        i = int(loc_pos[0])
        di = loc_pos[0] - i
        
        j = int(loc_pos[1])
        dj = loc_pos[1] - j
        
        k = int(loc_pos[2])
        dk = loc_pos[2] - k
        
        self.field[i,j,k] += val * (1 - di) * (1 - dj) * (1 - dk);
        self.field[i + 1,j,k] += val * (di) * (1 - dj) * (1 - dk);
        self.field[i + 1,j + 1,k] += val * (di) * (dj) * (1 - dk);
        self.field[i,j + 1,k] += val * (1 - di) * (dj) * (1 - dk);
        self.field[i,j,k + 1] += val * (1 - di) * (1 - dj) * (dk);
        self.field[i + 1,j,k + 1] += val * (di) * (1 - dj) * (dk);
        self.field[i + 1,j + 1,k + 1] += val * (di) * (dj) * (dk);
        self.field[i,j + 1,k + 1] += val * (1 - di) * (dj) * (dk);