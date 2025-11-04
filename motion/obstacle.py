from typing import Tuple, List, Dict, Optional
import numpy as np

class ObstacleMap:
    def __init__(self, safe_margin: float = 20) -> None:
        self.objects = {}
        self.safe_margin = safe_margin  # 안전 거리 설정

    def set_circle(self, name: str, centerx: float, centery: float, radius: float):
        o = {'type': 'circle', 'name': name, 'centerx': centerx, 'centery': centery, 'radius': radius}
        self.objects[name] = o

    def set_rectangle(self, name: str, centerx: float, centery: float, width: float, height: float, theta: float = 0.0):    
        o = {'type': 'rectangle', 'name': name, 'centerx': centerx, 'centery': centery, 
            'width': width, 'height': height, 'theta': theta}
        self.objects[name] = o

    def _distance_to_circle(self, x: float, y: float, obj: Dict) -> Tuple[float, float, float]:
        ox, oy, r = obj['centerx'], obj['centery'], obj['radius']
        dx, dy = x - ox, y - oy
        mag = np.sqrt(dx**2 + dy**2)
        
        if mag < 1e-10:  # 중심점에 있는 경우
            return -r, 0.0, 0.0
        
        distance = mag - r
        grad_x = dx / mag
        grad_y = dy / mag
        
        return distance, grad_x, grad_y

    def _distance_to_rectangle(self, x: float, y: float, obj: Dict) -> Tuple[float, float, float]:
        cx, cy = obj['centerx'], obj['centery']
        w, h = obj['width'], obj['height']
        theta = obj['theta']
        
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        
        dx = x - cx
        dy = y - cy
        
        local_x = dx * cos_theta + dy * sin_theta
        local_y = -dx * sin_theta + dy * cos_theta
        
        half_w = w / 2
        half_h = h / 2
        
        closest_x = np.clip(local_x, -half_w, half_w)
        closest_y = np.clip(local_y, -half_h, half_h)
        
        diff_x = local_x - closest_x
        diff_y = local_y - closest_y
        distance = np.sqrt(diff_x**2 + diff_y**2)
        
        if abs(local_x) <= half_w and abs(local_y) <= half_h:
            dist_to_edges = [
                half_w - abs(local_x),  
                half_h - abs(local_y) 
            ]
            distance = -min(dist_to_edges)
            
            if dist_to_edges[0] < dist_to_edges[1]:
                grad_local_x = np.sign(local_x)
                grad_local_y = 0
            else:
                grad_local_x = 0
                grad_local_y = np.sign(local_y)
        else:
            if distance < 1e-10:
                grad_local_x, grad_local_y = 0.0, 0.0
            else:
                grad_local_x = diff_x / distance
                grad_local_y = diff_y / distance
        
        grad_x = grad_local_x * cos_theta - grad_local_y * sin_theta
        grad_y = grad_local_x * sin_theta + grad_local_y * cos_theta
        
        return distance, grad_x, grad_y

    def get_d_grad(self, x: float, y: float) -> Tuple[float, float, float]:
        """
        가장 가까운 장애물까지의 거리와 그래디언트 반환
        
        Returns:
            (distance, grad_x, grad_y): 거리와 그래디언트 (거리가 증가하는 방향)
        """
        mindist = np.inf
        min_grad_x = 0.0
        min_grad_y = 0.0
        
        for o in self.objects.values():
            if o['type'] == 'circle':
                d, gx, gy = self._distance_to_circle(x, y, o)
            elif o['type'] == 'rectangle':
                d, gx, gy = self._distance_to_rectangle(x, y, o)
            else:
                continue
            
            if d < mindist:
                mindist = d
                min_grad_x = gx
                min_grad_y = gy
        
        if mindist == np.inf:
            return np.inf, 0.0, 0.0
        
        return mindist, min_grad_x, min_grad_y

    def cost(self, pos: np.ndarray) -> float:
        """
        위치에 대한 장애물 비용 계산
        
        Args:
            pos: 위치 벡터 [x, y, ...]
            
        Returns:
            비용 (0: 안전, 큰 값: 위험/충돌)
        """
        if len(pos) < 2:
            return 0.0
        
        x, y = pos[0], pos[1]
        distance, _, _ = self.get_d_grad(x, y)
        
        if distance >= self.safe_margin:
            return 0.0
        elif distance <= 0:
            return 1000.0  # 충돌
        else:
            # 거리가 가까울수록 비용 증가
            normalized_dist = distance / self.safe_margin  # [0, 1]
            cost = np.exp(-2 * normalized_dist) * 100
            return cost
    
    def is_collision(self, pos: np.ndarray, radius: float = 0.0) -> bool:
        """
        주어진 위치에서 충돌 여부 확인
        
        Args:
            pos: 위치 [x, y, ...]
            radius: 객체의 반경
        
        Returns:
            True if collision, False otherwise
        """
        if len(pos) < 2:
            return False
        
        x, y = pos[0], pos[1]
        distance, _, _ = self.get_d_grad(x, y)
        
        return distance < radius