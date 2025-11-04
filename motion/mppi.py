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

    def __init__(self, num_samples=128, lambda_=1.0, exploration_ratio=0.9, global_variance_scale=5.0):
        self.num_samples = num_samples
        self.best_traj = None
        self.trajectories = None
        self.gbp_weight_alpha = 0.5
        self.base_weight_beta = 2.0
        self.weights = None
        self.lambda_ = lambda_
        self.base_variance =10.0
        self.w_goal = 10.0
        self.w_obs = 50.0
        self.w_agent = 40.0
        self.w_smooth_vel = 15
        self.w_smooth_acc = 10
        self.trust_threshold = 2
        self.exploration_ratio = exploration_ratio
        self.global_variance_scale = global_variance_scale

    def sample_trajectories(self, means, covariances):
        num_nodes = len(means)

        first_mean = np.asarray(means[0]).flatten()
        state_dim = len(first_mean)

        samples = np.zeros((self.num_samples, num_nodes, state_dim))
        base_cov_component = np.eye(state_dim) * self.base_weight_beta * self.base_variance

        num_local = int(self.num_samples * self.exploration_ratio)
        num_global = self.num_samples - num_local

        for i in range(num_nodes):
            mean = np.asarray(means[i]).flatten()
            cov = np.asarray(covariances[i])
            
            if cov.shape != (state_dim, state_dim):
                print(f"Warning: cov shape {cov.shape} at node {i}, using identity")
                cov = np.eye(state_dim)

            gbp_cov_component = self.gbp_weight_alpha * cov
            combined_cov = gbp_cov_component + base_cov_component

            if np.trace(cov) < self.trust_threshold:
                node_samples = np.tile(mean, (self.num_samples, 1))
            else:
                local_samples = np.random.multivariate_normal(
                    mean=mean,
                    cov=combined_cov,
                    size=num_local
                )

                global_cov = np.eye(state_dim) * self.global_variance_scale * np.trace(combined_cov) / state_dim
                global_samples = np.random.multivariate_normal(
                    mean=mean,
                    cov=global_cov,
                    size=num_global
                )

                node_samples = np.vstack([local_samples, global_samples])
                np.random.shuffle(node_samples)

            samples[:, i, :] = node_samples

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
            # Env.find_near()를 통해 주변 agent만 고려
            near_agents = env.find_near(agent, range=500)
            for other in near_agents:
                other_pos = np.array([v.mean[:2, 0] for v in other._vnodes])  # (T, 2)
                for k in range(num_samples):
                    diff = pos[k] - other_pos[None, :, :]
                    dist = np.linalg.norm(diff, axis=-1)
                    safe_dist = r + other.r
                    penalty = np.exp(-0.5 * (dist / safe_dist)**2)
                    c_agent[k] += self.w_agent * penalty.flatten()

        vel = np.diff(pos, axis=1)  
        acc = np.diff(vel, axis=1)  
        
        c_smooth_vel = np.linalg.norm(vel, axis=-1)    
        c_smooth_acc = np.linalg.norm(acc, axis=-1)     

        # 차원 맞춰 합산
        c_smooth = np.zeros_like(dist_goal)
        c_smooth[:, 1:] += self.w_smooth_vel * c_smooth_vel
        c_smooth[:, 2:] += self.w_smooth_acc * c_smooth_acc

        total_costs = np.sum(c_goal + c_obs + c_agent + c_smooth, axis=1)  # (num_samples,)

        return total_costs