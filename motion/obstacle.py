from typing import Tuple, List, Dict
import numpy as np

class ObstacleMap:
    def __init__(self, safe_margin: float = 20) -> None:
        self.objects = {}
        self.safe_margin = safe_margin  # 안전 거리 설정

    def set_circle(self, name: str, centerx, centery, radius):
        o = {'type': 'circle', 'name': name, 'centerx': centerx, 'centery': centery, 'radius': radius}
        self.objects[name] = o

    def set_rectangle(self, name: str, centerx: float, centery: float, width: float, height: float, theta: float = 0.0):    
        o = {'type': 'rectangle', 'name': name, 'centerx': centerx, 'centery': centery, 
            'width': width, 'height': height, 'theta': theta}
        self.objects[name] = o

    def get_d_grad(self, x, y) -> Tuple[float, float, float]:
        mindist = np.inf
        mino = None
        for o in self.objects.values():
            if o['type'] == 'circle':
                ox, oy, r = o['centerx'], o['centery'], o['radius']
                d = np.sqrt((x - ox)**2 + (y - oy)**2) - r
                if d < mindist:
                    mindist = d
                    mino = o
        if mino is None:
            return np.inf, 0, 0
        if mino['type'] == 'circle':
            ox, oy = o['centerx'], o['centery']
            dx, dy = x - ox, y - oy
            mag = np.sqrt(dx**2 + dy**2)
            return mindist, dx/mag, dy/mag

    def cost(self, pos: np.ndarray) -> float:

        if len(pos) >= 2:
            x, y = pos[0], pos[1]
        else:
            return 0.0
        
        distance, _, _ = self.get_d_grad(x, y)
        
        if distance >= self.safe_margin:
            return 0.0
        elif distance <= 0:
            return 1000.0
        else:
            normalized_dist = distance / self.safe_margin  # [0, 1]
            cost = np.exp(-2 * normalized_dist) * 100
            
            return cost
    
    def is_collision(self, pos: np.ndarray, radius: float = 0.0) -> bool:
        """
        주어진 위치에서 충돌 여부 확인
        
        Args:
            pos: 위치 [x, y]
            radius: 객체의 반경
        
        Returns:
            True if collision, False otherwise
        """
        if len(pos) < 2:
            return False
        
        x, y = pos[0], pos[1]
        distance, _, _ = self.get_d_grad(x, y)
        
        return distance < radius