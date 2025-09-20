# save_dat_to_hdf5_simple.py
import numpy as np
import pandas as pd
import h5py
import os


dat_file_mag = "b_3d.dat"
dat_file_exc = "excitation.dat"
dat_file_ion = "ionization.dat"
cols_mag = ["x","y","z","bx","by","bz","b"]
cols_exc = ["energy","1","2","3","4","5","6","7","8","9"]
cols_ion = ["energy","1","2","3","4","5","6","7","8","9","10","11","12","13"]

h5_file = "data.h5"

# ------ 磁场网格参数 -----
nx_b = 201; ny_b = 201; nz_b = 423
xmin, ymin, zmin = -0.0625, -0.0625, 0
xmax, ymax, zmax = 0.0625, 0.0625, 0.422
dx = (xmax - xmin) / (nx_b - 1)
dy = (ymax - ymin) / (ny_b - 1)
dz = (zmax - zmin) / (nz_b - 1)
tol = 1e-6   # 坐标到格点的容差，按需要调整
# ------------------------

def coords_to_indices(x, xmin, dx):
    """将坐标映射到索引，使用四舍五入并返回 int 索引"""
    idx = np.rint((x - xmin) / dx).astype(np.int64)
    return idx

# 统计行数（快）
def count_lines(fname):
    with open(fname, "rb") as f:
        return sum(1 for _ in f)



# 创建 HDF5 dataset
dtype = np.float64         # 可改为 float64 若需精度
chunksize = nx_b * ny_b        # 每个 chunk 的行数（按需调整）

with h5py.File(h5_file, "w") as f:
    nrows = count_lines(dat_file_mag)
    print("rows:", nrows)
    dset = f.create_dataset(
        "magnetic_fields",
        shape=(nx_b, ny_b, nz_b, 3),
        dtype=dtype,
        chunks=(nx_b, ny_b, 1, 3),
        compression="gzip",        # 或 "lzf"（更快但压缩率低）
        compression_opts=4
    )
    # 用 pandas 流式读取并写入
    reader = pd.read_csv(
        dat_file_mag, sep=r"\s+", header=None, names=cols_mag,
        dtype=dtype, chunksize=chunksize, engine="c"
    )
    
    
    # idx = 0
    # for chunk in reader:
    #     arr = chunk.values  # numpy array shape (m,7)
    #     m = arr.shape[0]
    #     dset[idx: idx + m, :] = arr
    #     idx += m
    #     if idx % (chunksize*10) == 0:
    #         print("written rows:", idx)
    
    iz = 0
    total =0
    for chunk in reader:
        xs = chunk["x"].to_numpy()
        ys = chunk["y"].to_numpy()
        zs = chunk["z"].to_numpy()
        vals = chunk[["bx","by","bz"]].to_numpy().astype(dtype)
        # 计算索引
        ix = coords_to_indices(xs, xmin, dx)
        iy = coords_to_indices(ys, ymin, dy)
        iz = coords_to_indices(zs, zmin, dz)
        
        # 精度检查（坐标是否在容差内对齐到网格）
        # 计算重构坐标与原坐标差值
        x_recon = xmin + ix * dx
        y_recon = ymin + iy * dy
        z_recon = zmin + iz * dz
        bad_mask = (~np.isclose(xs, x_recon, atol=tol)) | (~np.isclose(ys, y_recon, atol=tol)) | (~np.isclose(zs, z_recon, atol=tol))
        if np.any(bad_mask):
            # 可选择改用更宽容的 rtol/atol 或用最近邻映射
            bad_count = np.count_nonzero(bad_mask)
            raise ValueError(f"{bad_count} 行的坐标未对齐到规则网格（超出容差 tol={tol}）。查看样例行:\n{chunk[bad_mask].head()}")

        # 边界检查
        oob = (ix < 0) | (ix >= nx_b) | (iy < 0) | (iy >= ny_b) | (iz < 0) | (iz >= nz_b)
        if np.any(oob):
            raise IndexError(f"有 {np.count_nonzero(oob)} 个点索引越界。请检查 xmin/dx 或 nx_b/ny_b/nz_b。样例越界行:\n{chunk[oob].head()}")
        # 为健壮可检查
        unique_iz = np.unique(iz)
        if unique_iz.size != 1:
            raise ValueError("当前块不只属于一个 iz, 确认 chunksize 是否等于 nx*ny")
        # reshape vals 按 (nx, ny, 3) 并写入 dset[:, :, iz, :]
        slab = vals.reshape((nx_b, ny_b, 3), order="C")
        dset[:, :, unique_iz[0], :] = slab

        total += len(chunk)
        if total % (chunksize * 5) == 0:
            print(f"written {total} rows")
    print("写入完成，总行数:", total)        
    
    print("mag read finished")
            
    
    nrows = count_lines(dat_file_exc)    
    dset = f.create_dataset (
        "excitation",
        shape=(nrows, 10),
        dtype=dtype,
        chunks=(min(chunksize, nrows), 10),
        compression="gzip",        # 或 "lzf"（更快但压缩率低）
        compression_opts=4
    )
    
    reader = pd.read_csv(
        dat_file_exc, sep=r"\s+", header=None, names=cols_exc,
        dtype=dtype, chunksize=chunksize, engine="c"
    )
    idx = 0
    for chunk in reader:
        arr = chunk.values  # numpy array shape (m,7)
        m = arr.shape[0]
        dset[idx: idx + m, :] = arr
        idx += m
        if idx % (chunksize*10) == 0:
            print("written rows:", idx)
    print("excitation read finished")
    
    
    
    nrows = count_lines(dat_file_ion)    
    dset = f.create_dataset (
        "ionization",
        shape=(nrows, 14),
        dtype=dtype,
        chunks=(min(chunksize, nrows), 14),
        compression="gzip",        # 或 "lzf"（更快但压缩率低）
        compression_opts=4
    )
    
    reader = pd.read_csv(
        dat_file_ion, sep=r"\s+", header=None, names=cols_ion,
        dtype=dtype, chunksize=chunksize, engine="c"
    )
    idx = 0
    for chunk in reader:
        arr = chunk.values  # numpy array shape (m,7)
        m = arr.shape[0]
        dset[idx: idx + m, :] = arr
        idx += m
        if idx % (chunksize*10) == 0:
            print("written rows:", idx)
    print("ionization read finished")

print("done")