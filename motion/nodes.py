from typing import Tuple, List, Dict
import numpy as np
from fg.gaussian import Gaussian
from fg.factor_graph import VNode, FNode, FactorGraph

from .obstacle import ObstacleMap


class RemoteVNode(VNode):
    def __init__(self, name: str, dims: list, belief: Gaussian = None) -> None:
        super().__init__(name, dims, belief)
        self._msgs = {}

    def update_belief(self) -> Gaussian:
        # return super().update_belief()
        return None
    def calc_msg(self, edge):
        # return super().calc_msg(edge)
        return self._msgs.get(edge, None)


class DynaFNode(FNode):
    def __init__(self, name: str, vnodes: List[VNode], factor: Gaussian = None,
                 dt: float = 0.1, sigma_acc: float = 0.15, eps_Q: float = 1e-1, eps_prec: float = 1e-6) -> None:
        assert len(vnodes) == 2
        super().__init__(name, vnodes, factor)
        self._dt = dt
        self._sigma_acc = sigma_acc
        self._eps_Q = eps_Q
        self._eps_prec = eps_prec

    def update_factor(self):
        # NOTE target: ||h(x) - z)||2 -> 0
        dt = self._dt
        v0 = self._vnodes[0].mean
        v1 = self._vnodes[1].mean
        v = np.vstack([v0, v1])  # [8, 1]
        z = np.zeros((4, 1))

        # kinetic
        k = np.identity(4)   # [4, 4]
        k[:2, 2:] = np.identity(2) * dt

        h = k @ v0 - v1     # [4, 1]
        # jacob of h
        jacob = np.array([
            [1, 0, dt, 0, -1, 0, 0, 0],  # h(x)[0] = dx = x(k) + vx(k) * dt - x(k+1)
            [0, 1, 0, dt, 0, -1, 0, 0],  # h(x)[1] = dy = y(k) + vy(k) * dt - y(k+1)
            [0, 0, 1, 0, 0, 0, -1, 0],  # h(x)[2] = dvx = vx(k) - vx(k+1)
            [0, 0, 0, 1, 0, 0, 0, -1],  # h(x)[3] = dvy = vy(k) - vy(k+1)
        ])  # [4, 8]

        # presicion of observation z (i.e. the target of h) (here is zero)
        q11 = dt**3 / 3.0
        q12 = dt**2 / 2.0
        q22 = dt
        Q_axis = (self._sigma_acc**2) * np.array([[q11, q12], [q12, q22]])  # 2x2

        # build 4x4 observation covariance for [dx, dy, dvx, dvy]
        Q_obs = np.block([
            [Q_axis, np.zeros((2,2))],
            [np.zeros((2,2)), Q_axis]
        ])  # 4x4

        # numeric stabilization
        Q_obs += self._eps_Q * np.eye(4)
        # observation precision
        precision = np.linalg.inv(Q_obs)

        # joint information
        prec = jacob.T @ precision @ jacob  # 8x8
        # final regularization to ensure PD
        prec += self._eps_prec * np.eye(8)

        info = jacob.T @ precision @ (jacob @ v + z - h)

        self._factor = Gaussian.from_info(self.dims, info, prec)


class ObstacleFNode(FNode):
    def __init__(self, name: str, vnodes: List[VNode], factor: Gaussian = None,
                 omap: ObstacleMap = None, safe_dist: float = 5, z_precision: float = 100) -> None:
        assert len(vnodes) == 1
        super().__init__(name, vnodes, factor)
        self._omap = omap
        self._safe_dist = safe_dist
        self._z_precision = z_precision

    def update_factor(self):
        # target: ||h(x) - z)||2 -> 0
        z = np.zeros((1, 1))
        v = self._vnodes[0].mean  # [4, 1]

        distance, distance_gradx, distance_grady = self._omap.get_d_grad(v[0, 0], v[1, 0])
        # distance -= self._safe_radius

        h = np.array([[max(0, 1 - distance / self._safe_dist)]])
        if distance > self._safe_dist:
            jacob = np.zeros((1, 4))
        else:
            jacob = np.array([[-distance_gradx/self._safe_dist, -distance_grady/self._safe_dist, 0, 0]])  # [1, 4]
        precision = np.identity(1) * self._z_precision * self._safe_dist**2

        prec = jacob.T @ precision @ jacob
        info = jacob.T @ precision @ (jacob @ v + z - h)
        self._factor = Gaussian.from_info(self.dims, info, prec)


class DistFNode(FNode):
    def __init__(self, name: str, vnodes: List[VNode], factor: Gaussian = None,
                 safe_dist: float = 20, z_precision: float = 100) -> None:
        assert len(vnodes) == 2
        super().__init__(name, vnodes, factor)
        self._safe_dist = safe_dist
        self._z_precision = z_precision

    def update_factor(self):
        # target: ||h(x) - z)||2 -> 0
        z = np.zeros((1, 1))
        v0 = self._vnodes[0].mean  # [4, 1]
        v1 = self._vnodes[1].mean  # [4, 1]
        if np.allclose(v0, v1):
            v1 += np.random.rand(4, 1) * 0.01
        v = np.vstack([v0, v1])  # [8, 1]

        distance = np.linalg.norm(v0[:2, 0] - v1[:2, 0])
        distance_gradx0, distance_grady0 = v0[0, 0]-v1[0, 0], v0[1, 0]-v1[1, 0]
        distance_gradx0 /= distance
        distance_grady0 /= distance

        if distance > self._safe_dist:
            prec = np.identity(8) * 0.0001
            info = prec @ v

        else:
            h = np.array([[1 - distance / self._safe_dist]])
            jacob = np.array([[
                -distance_gradx0/self._safe_dist, -distance_grady0/self._safe_dist, 0, 0,
                distance_gradx0/self._safe_dist, distance_grady0/self._safe_dist, 0, 0]])  # [1, 8]
            precision = np.identity(1) * self._z_precision * (self._safe_dist**2)

            prec = jacob.T @ precision @ jacob
            info = jacob.T @ precision @ (jacob @ v + z - h)

            # NOTE
            # prec has a structure like [A, 0, C, 0] , which is not invertable
            #                           [0, 0, 0, 0]
            #                           [D, 0, B, 0]
            #                           [0, 0, 0, 0]
            # modify prec to be [A, 0, C, 0] to avoid this problem and will not affect result.
            #                   [0, I, 0, 0]
            #                   [D, 0, B, 0]
            #                   [0, 0, 0, I]
            prec[2, 2] = 1
            prec[3, 3] = 1
            prec[6, 6] = 1
            prec[7, 7] = 1
        self._factor = Gaussian.from_info(self.dims, info, prec)

class MPPIFNode(FNode):
    def __init__(self, name: str, vnodes: List[VNode], factor: Gaussian = None,
                 mppi_traj: np.ndarray = None, mppi_precision: float = 5.0) -> None:
        assert len(vnodes) == 1
        super().__init__(name, vnodes, factor)
        self._mppi_traj = mppi_traj
        self._mppi_precision = mppi_precision
        
    def set_mppi_trajectory(self, traj: np.ndarray):
        self._mppi_traj = traj
        
    def update_factor(self):
        if self._mppi_traj is None:
            # MPPI 궤적이 없으면 identity factor
            self._factor = Gaussian.identity(self.dims)
            return
            
        v = self._vnodes[0].mean  # [4, 1]
        z = self._mppi_traj  # [4, 1] - MPPI가 제안한 목표 state
        
        # target: ||h(x) - z)||^2 -> 0
        # h(x) = x (identity mapping)
        h = v
        jacob = np.identity(4)  # [4, 4]
        
        # MPPI 궤적을 따르는 정도 조절
        # position은 더 강하게, velocity는 더 약하게 따를 수도 있음
        precision = np.diag([
            self._mppi_precision,      # x
            self._mppi_precision,      # y  
            self._mppi_precision * 0.5,  # vx (속도는 덜 제약)
            self._mppi_precision * 0.5   # vy
        ])
        
        prec = jacob.T @ precision @ jacob
        info = jacob.T @ precision @ (jacob @ v + z - h)
        
        self._factor = Gaussian.from_info(self.dims, info, prec)