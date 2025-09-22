import numpy as np
import h5py

class BaseField:
    def __init__(self, nn):
        self.nn = nn

    def setBox(self, box_min, box_max):
        self.box_min = box_min
        self.box_max = box_max
        self.dh = (box_max - box_min) / (self.nn - 1)
    
    def XToL(self, pos) -> np.array:
        lc = np.zeros(3)
        lc = (pos - self.box_min) / self.dh
        return lc

    def gather(self, pos): 
        loc_pos = self.XToL(pos)
        i = int(loc_pos[0])
        di = loc_pos[0] - i
        
        j = int(loc_pos[1])
        dj = loc_pos[1] - j
        
        k = int(loc_pos[2])
        dk = loc_pos[2] - k
        
        val = self.field[i,j,k] * (1 - di) * (1 - dj) * (1 - dk) + \
              self.field[i + 1,j,k] * (di) * (1 - dj) * (1 - dk) + \
              self.field[i + 1,j + 1,k] * (di) * (dj) * (1 - dk) + \
              self.field[i,j + 1,k] * (1 - di) * (dj) * (1 - dk) + \
              self.field[i,j,k + 1] * (1 - di) * (1 - dj) * (dk) + \
              self.field[i + 1,j,k + 1] * (di) * (1 - dj) * (dk) + \
              self.field[i + 1,j + 1,k + 1] * (di) * (dj) * (dk) + \
              self.field[i,j + 1,k + 1] * (1 - di) * (dj) * (dk)
        return val
    
    def scatter(self,pos,val):
        loc_pos = self.XToL(pos)
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

class Field(BaseField):
    def __init__(self, nn):
        super(Field,self).__init__(nn)
        
    def load(self, filename, dataset_name):
        with h5py.File(filename, "r") as f:
            self.field = f[dataset_name][...]
        return self.field
    
