import numpy as np
from fg.gaussian import Gaussian


class GBPMPPI:
    """
    Monte Carlo Path Integral Controller driven by GBP-inferred beliefs.
    Each agent has its own MPPI instance.

    Attributes
    ----------
    num_samples : int
        Number of trajectories to sample
    lambda_ : float
        Temperature parameter controlling softmin weighting
    """

    def __init__(self, num_samples=128, lambda_=1.0):
        self.num_samples = num_samples
        self.best_traj = None
        self.trajectories = None
        self.weights = None
        self.lambda_ = lambda_
        self.w_goal = 1.0
        self.w_obs = 5.0
        self.w_agent = 5.0

    def sample_trajectories(self, means, covariances):
        """
        각 변수 노드(mean, cov)에 대해 Monte Carlo 샘플링을 수행
        
        Args:
            means: List of mean vectors, each can be shape [4,] or [4, 1]
            covariances: List of covariance matrices, each shape [4, 4]
        
        Returns:
            samples: np.ndarray of shape (num_samples, num_nodes, state_dim)
        """
        num_nodes = len(means)
        
        # 첫 번째 mean으로부터 state dimension 추출
        first_mean = np.asarray(means[0]).flatten()
        state_dim = len(first_mean)
        
        samples = np.zeros((self.num_samples, num_nodes, state_dim))
        
        for i in range(num_nodes):
            # mean을 1D 배열로 변환 ([4, 1] -> [4,])
            mean = np.asarray(means[i]).flatten()
            cov = np.asarray(covariances[i])
            
            # covariance가 올바른 shape인지 확인
            if cov.shape != (state_dim, state_dim):
                print(f"Warning: cov shape {cov.shape} at node {i}, using identity")
                cov = np.eye(state_dim)
            
            # Sampling
            samples[:, i, :] = np.random.multivariate_normal(
                mean=mean,
                cov=cov,
                size=self.num_samples
            )
        
        self.trajectories = samples
        return samples

    def compute_weights(self, costs):
        """
        Compute importance weights using softmin
        w_i = exp(-cost_i / lambda) / sum_j(exp(-cost_j / lambda))
        
        Args:
            costs: np.ndarray of shape (num_samples,)
        
        Returns:
            weights: np.ndarray of shape (num_samples,)
        """
        # Numerical stability: subtract min cost
        min_cost = np.min(costs)
        weights = np.exp(-(costs - min_cost) / self.lambda_)
        
        # Normalize
        weights /= np.sum(weights)
        
        self.weights = weights
        return weights

    def integrate_path(self, weights=None, trajectories=None):
        """
        Compute weighted average trajectory
        
        Args:
            weights: Optional, use stored weights if None
            trajectories: Optional, use stored trajectories if None
        
        Returns:
            best_traj: List of np.ndarray, each shape [state_dim, 1]
        """
        if weights is None:
            weights = self.weights
        if trajectories is None:
            trajectories = self.trajectories
            
        if trajectories is None or weights is None:
            raise ValueError("Trajectories or weights not computed.")
        
        # Weighted average: (num_samples,) · (num_samples, horizon, state_dim)
        weighted_traj = np.tensordot(weights, trajectories, axes=(0, 0))
        # weighted_traj shape: (horizon, state_dim)
        
        # Convert to list of [state_dim, 1] arrays for compatibility
        self.best_traj = [weighted_traj[i:i+1, :].T for i in range(weighted_traj.shape[0])]
        
        return self.best_traj

    def cost_func(self, trajs: np.ndarray, agent) -> np.ndarray:
        """
        각 trajectory별 total cost 계산
        
        Args:
            trajs (np.ndarray): shape (num_samples, horizon, state_dim)
            agent (Agent): 현재 agent (env와 omap 접근 가능)
        
        Returns:
            np.ndarray: shape (num_samples,)
        """
        env = agent._env
        omap = agent._omap
        target = agent.get_target()[:2, 0]  # [2,]
        r = agent.r
        num_samples, horizon, state_dim = trajs.shape

        # Position extraction (x, y)
        pos = trajs[:, :, :2]   # (num_samples, horizon, 2)
        
        # 1. Goal cost: distance to target
        dist_goal = np.linalg.norm(pos - target[None, None, :], axis=-1)  # (num_samples, horizon)
        c_goal = self.w_goal * dist_goal

        # 2. Obstacle cost
        if omap is not None:
            c_obs = np.zeros_like(dist_goal)
            for k in range(num_samples):
                for t in range(horizon):
                    c_obs[k, t] = omap.cost(pos[k, t])
            c_obs *= self.w_obs
        else:
            c_obs = 0

        # 3. Agent collision cost
        c_agent = np.zeros_like(dist_goal)
        if env is not None:
            for other in env._agents:
                if other is agent:
                    continue
                
                # Get other agent's planned trajectory
                other_states = other.get_state()
                if other_states[0] is None:
                    continue
                
                # Extract positions from other agent's trajectory
                other_pos = []
                for state in other_states:
                    if state is None:
                        break
                    other_pos.append(state[:2])  # x, y
                
                if len(other_pos) == 0:
                    continue
                
                other_pos = np.array(other_pos)  # (horizon, 2)
                
                # Compute collision cost for each sample
                safe_dist = r + other.r
                for k in range(num_samples):
                    for t in range(min(horizon, len(other_pos))):
                        # Distance between this trajectory and other's trajectory
                        dist = np.linalg.norm(pos[k, t] - other_pos[t])
                        
                        # Penalty increases as distance decreases
                        if dist < safe_dist * 2:  # Only penalize when close
                            penalty = np.exp(-0.5 * (dist / safe_dist)**2)
                            c_agent[k, t] += self.w_agent * penalty

        # Total cost: sum over time horizon
        total_costs = np.sum(c_goal + c_obs + c_agent, axis=1)  # (num_samples,)

        return total_costs