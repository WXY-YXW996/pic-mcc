import numpy as np
import numba
import Field
import fenics
    

class World(Field.BaseField):
    def __init__(self, nn: int, box_min, box_max) -> None:
        """
        初始化World对象

        参数:
            nn: int - 网格点数
        """
        super(World, self).__init__(nn)

        self.time: float = 0.0
        self.ts: int = 0
        self.dt: float = 0.0
        self.setBox(box_min, box_max)
        self.rho = Field.Field(self.nn)
        self.rho.setBox(box_min, box_max)
        self.rho.initializeField(n_components=1)
        self.potential = Field.Field(self.nn)
        self.potential.setBox(box_min, box_max)
        self.potential.initializeField(n_components=1)

    def setTime(self, dt: float, num_ts: int) -> None:
        """
        设置时间步长和总时间步数

        参数:
            dt: float - 时间步长
            num_ts: int - 总时间步数
        """
        self.dt: float = dt
        self.num_ts: int = num_ts

    def setFre(self, fre: float) -> None:
        """
        设置微波频率

        参数:
            fre: float - 微波频率
        """
        self.fre = fre
        self.b_ecr = fre / 28e9  # ECR磁场强度

    def advanceTime(self) -> bool:
        """
        推进时间一步

        返回:
            bool - 如果仍在模拟时间范围内返回True, 否则返回False
        """
        self.time += self.dt
        self.ts += 1
        return self.ts <= self.num_ts

    def inbound(self, pos: np.array) -> bool:
        """
        检查位置是否在世界边界内

        参数:
            pos: np.array - 位置坐标数组

        返回:
            bool - 如果位置在边界内返回True，否则返回False
        """
        # 使用np.any检查是否有任何坐标超出边界
        if np.any(pos > self.box_max) or np.any(pos < self.box_min):
            return False
        return True

    def computeRho(self, species_list):
        """
        计算电荷密度

        参数:
            species_list: list - 粒子种类列表
        """
        for sp in species_list:
            self.rho.field += sp.den.field * sp.charge * sp.weight

        # computeChargeRho(self.rho, species_list)

    def potentialSlover(self, solver_type='cg', preconditioner='ilu', 
                        max_iterations=10000, tolerance=1e-6):
        """
        根据可用的库版本选择求解器
        """

        return self._solve_with_fenics(solver_type, preconditioner, max_iterations, tolerance)
    
    
    def _solve_with_fenics(self, solver_type='gmres', preconditioner='jacobi', 
                          max_iterations=10000, tolerance=1e-6):
        """
        使用传统 FEniCS 库求解3D泊松方程: ∇²φ = -ρ/ε₀ (向后兼容)
        """
        # 物理常数
        epsilon_0 = 8.854e-12  # 真空介电常数 F/m
        
        # 获取网格参数
        nx, ny, nz = self.nn
        x_min, y_min, z_min = self.box_min
        x_max, y_max, z_max = self.box_max
        
        mesh_nx, mesh_ny, mesh_nz = nx-1, ny-1, nz-1
        print(f"创建传统 FEniCS 网格: {mesh_nx+1} × {mesh_ny+1} × {mesh_nz+1}")
        
        # 创建FEniCS网格
        domain = fenics.BoxMesh(
            fenics.Point(x_min, y_min, z_min),
            fenics.Point(x_max, y_max, z_max),
            mesh_nx, mesh_ny, mesh_nz
        )
        
        # 定义函数空间（使用一阶拉格朗日元）
        V = fenics.FunctionSpace(domain, 'P', 1)
        
        # 定义边界条件（这里使用齐次狄利克雷边界条件：φ=0 在边界上）
        def boundary(x, on_boundary):
            return on_boundary
        
        u_D = fenics.Constant(0.0)
        bc = fenics.DirichletBC(V, u_D, boundary)
        
        # 将网格电荷密度插值到FEniCS函数空间
        rho_func = self._interpolate_rho_to_fenics(V)
        
        # 定义变分问题
        u = fenics.TrialFunction(V)
        v = fenics.TestFunction(V)
        
        # 泊松方程的弱形式: ∫∇u·∇v dx = ∫(-ρ/ε₀)v dx
        a = fenics.dot(fenics.grad(u), fenics.grad(v)) * fenics.dx
        L = -(rho_func / epsilon_0) * v * fenics.dx
        
        # 求解线性系统
        u_solution = fenics.Function(V)
        
        print(f"使用传统 FEniCS 迭代求解器: {solver_type} with {preconditioner} 预条件器")
            
        # 设置求解器参数
        solver_params = {
            'linear_solver': 'gmres',
            'preconditioner': 'jacobi',
            'krylov_solver': {
                'maximum_iterations': max_iterations * 2,
                'absolute_tolerance': tolerance * 10,
                'relative_tolerance': tolerance * 10,
                'monitor_convergence': True
            }
        }

        fenics.solve(a == L, u_solution, bc, solver_parameters=solver_params)
                
        # 将解插值回规则网格
        self._interpolate_solution_to_grid(u_solution)
        
        # 计算电场 E = -∇φ
        self._compute_electric_field_from_potential()
    
    def _interpolate_rho_to_fenics(self, V):
        """
        将规则网格上的电荷密度插值到FEniCS函数空间
        
        参数:
            V: FEniCS函数空间
            
        返回:
            fenics.Function: 插值后的电荷密度函数
        """
        # 创建FEniCS函数
        rho_func = fenics.Function(V)
        
        # 获取函数空间的自由度坐标
        dofs_coordinates = V.tabulate_dof_coordinates()
        
        # 对每个自由度点，从规则网格插值电荷密度
        rho_values = np.zeros(len(dofs_coordinates))
        
        for i, coord in enumerate(dofs_coordinates):
            # 将物理坐标转换为局部网格坐标
            loc_pos = self.XToL(coord)
            
            # 检查是否在网格范围内
            if (loc_pos >= 0).all() and (loc_pos <= np.array(self.nn) - 1).all():
                # 使用三线性插值获取电荷密度值
                rho_values[i] = self.rho.gather(loc_pos)[0]  # 取第一个分量
            else:
                rho_values[i] = 0.0  # 网格外设为0
        
        # 设置函数值
        rho_func.vector()[:] = rho_values
        
        return rho_func
    
    def _interpolate_solution_to_grid(self, u_solution):
        """
        将FEniCS解插值回规则网格
        
        参数:
            u_solution: FEniCS函数，泊松方程的解
        """
        # 清零电势场
        self.potential.clear()
        
        # 遍历规则网格的每个点
        for i in range(self.nn[0]):
            for j in range(self.nn[1]):
                for k in range(self.nn[2]):
                    # 计算物理坐标
                    x = self.box_min[0] + i * self.dh[0]
                    y = self.box_min[1] + j * self.dh[1]
                    z = self.box_min[2] + k * self.dh[2]
                    
                    try:
                        # 在该点评估FEniCS函数
                        phi_value = u_solution(x, y, z)
                        self.potential.field[i, j, k, 0] = phi_value
                    except:
                        # 如果点在网格外，设为0
                        self.potential.field[i, j, k, 0] = 0.0
    
    def _compute_electric_field_from_potential(self):
        """
        从电势计算电场: E = -∇φ
        使用中心差分法计算梯度
        """
        # 确保电场数组已初始化
        if not hasattr(self, 'electric_field'):
            self.electric_field = Field.Field(self.nn)
            self.electric_field.setBox(self.box_min, self.box_max)
            self.electric_field.initializeField(n_components=3)
        
        self.electric_field.clear()
        
        # 计算电场的每个分量
        nx, ny, nz = self.nn
        
        for i in range(1, nx-1):
            for j in range(1, ny-1):
                for k in range(1, nz-1):
                    # Ex = -dφ/dx
                    self.electric_field.field[i, j, k, 0] = -(
                        self.potential.field[i+1, j, k, 0] - 
                        self.potential.field[i-1, j, k, 0]
                    ) / (2 * self.dh[0])
                    
                    # Ey = -dφ/dy
                    self.electric_field.field[i, j, k, 1] = -(
                        self.potential.field[i, j+1, k, 0] - 
                        self.potential.field[i, j-1, k, 0]
                    ) / (2 * self.dh[1])
                    
                    # Ez = -dφ/dz
                    self.electric_field.field[i, j, k, 2] = -(
                        self.potential.field[i, j, k+1, 0] - 
                        self.potential.field[i, j, k-1, 0]
                    ) / (2 * self.dh[2])
        
        # 处理边界上的电场（使用向前/向后差分）
        self._compute_boundary_electric_field()
    
    def _compute_boundary_electric_field(self):
        """
        计算边界上的电场，使用向前/向后差分
        """
        nx, ny, nz = self.nn
        
        # 处理 x 方向边界
        for j in range(ny):
            for k in range(nz):
                # x=0 边界（使用向前差分）
                if nx > 1:
                    self.electric_field.field[0, j, k, 0] = -(
                        self.potential.field[1, j, k, 0] - 
                        self.potential.field[0, j, k, 0]
                    ) / self.dh[0]
                    
                    # x=nx-1 边界（使用向后差分）
                    self.electric_field.field[nx-1, j, k, 0] = -(
                        self.potential.field[nx-1, j, k, 0] - 
                        self.potential.field[nx-2, j, k, 0]
                    ) / self.dh[0]
        
        # 处理 y 方向边界
        for i in range(nx):
            for k in range(nz):
                # y=0 边界
                if ny > 1:
                    self.electric_field.field[i, 0, k, 1] = -(
                        self.potential.field[i, 1, k, 0] - 
                        self.potential.field[i, 0, k, 0]
                    ) / self.dh[1]
                    
                    # y=ny-1 边界
                    self.electric_field.field[i, ny-1, k, 1] = -(
                        self.potential.field[i, ny-1, k, 0] - 
                        self.potential.field[i, ny-2, k, 0]
                    ) / self.dh[1]
        
        # 处理 z 方向边界
        for i in range(nx):
            for j in range(ny):
                # z=0 边界
                if nz > 1:
                    self.electric_field.field[i, j, 0, 2] = -(
                        self.potential.field[i, j, 1, 0] - 
                        self.potential.field[i, j, 0, 0]
                    ) / self.dh[2]
                    
                    # z=nz-1 边界
                    self.electric_field.field[i, j, nz-1, 2] = -(
                        self.potential.field[i, j, nz-1, 0] - 
                        self.potential.field[i, j, nz-2, 0]
                    ) / self.dh[2]


