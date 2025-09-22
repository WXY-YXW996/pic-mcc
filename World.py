import numpy as np
import Field

class World(Field.Field):
    def __init__(self, nn: int) -> None:
        """
        初始化World对象
        
        参数:
            nn: int - 网格点数
        """
        super(World,self).__init__(nn)
        
        self.time:float = 0.0
        self.ts:int = 0

    def setTime(self, dt: float, num_ts: int) -> None:
        """
        设置时间步长和总时间步数
        
        参数:
            dt: float - 时间步长
            num_ts: int - 总时间步数
        """
        self.dt:float = dt
        self.num_ts:int = num_ts
    
    def advanceTime(self) -> bool:
        """
        推进时间一步
        
        返回:
            bool - 如果仍在模拟时间范围内返回True, 否则返回False
        """
        self.time +=self.dt
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
