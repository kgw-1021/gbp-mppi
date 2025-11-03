from typing import Tuple, List, Dict
import numpy as np

class ObstacleMap:
    def __init__(self, safe_margin: float = 20) -> None:
        self.objects = {}
        self.safe_margin = safe_margin  # 안전 거리 설정

    def set_circle(self, name: str, centerx, centery, radius):
        o = {'type': 'circle', 'name': name, 'centerx': centerx, 'centery': centery, 'radius': radius}
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
        """
        주어진 위치에서의 장애물 비용 계산
        
        Args:
            pos: 위치 [x, y] 또는 [x, y, vx, vy]
        
        Returns:
            cost: 장애물로부터의 거리 기반 비용
        """
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
    
    def cost_with_gradient(self, pos: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        비용과 그래디언트를 함께 반환 (최적화에 유용)
        
        Args:
            pos: 위치 [x, y]
        
        Returns:
            cost: 장애물 비용
            gradient: 비용의 그래디언트 [grad_x, grad_y]
        """
        if len(pos) < 2:
            return 0.0, np.zeros(2)
        
        x, y = pos[0], pos[1]
        distance, grad_x, grad_y = self.get_d_grad(x, y)
        
        if distance >= self.safe_margin:
            return 0.0, np.zeros(2)
        elif distance <= 0:
            # 장애물 내부: 밖으로 나가는 방향으로 큰 gradient
            return 1000.0, np.array([-grad_x * 1000, -grad_y * 1000])
        else:
            normalized_dist = distance / self.safe_margin
            cost = np.exp(-2 * normalized_dist) * 100
            
            # 비용의 gradient: chain rule 적용
            # d(cost)/d(pos) = d(cost)/d(distance) * d(distance)/d(pos)
            dcost_ddist = -200 / self.safe_margin * np.exp(-2 * normalized_dist)
            gradient = np.array([dcost_ddist * grad_x, dcost_ddist * grad_y])
            
            return cost, gradient
    
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