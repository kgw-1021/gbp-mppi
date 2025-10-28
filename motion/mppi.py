import numpy as np
from fg.gaussian import Gaussian
from .agent import Agent

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

    def __init__(self, num_samples=128):
        self.num_samples = num_samples
        self.best_traj = None
        self.trajectories = None
        self.weights = None
        self.lambda_ = 1.0
        self.w_goal = 1.0
        self.w_obs = 5.0
        self.w_agent = 5.0

    def sample_trajectories(self, means, covariances):
        """
        각 변수 노드(mean, cov)에 대해 Monte Carlo 샘플링을 수행
        means: [N, dim]
        covariances: [N, dim, dim]
        """
        num_nodes = len(means)
        samples = np.zeros((self.num_samples, num_nodes, means[0].shape[0]))
        for i in range(num_nodes):
            samples[:, i, :] = np.random.multivariate_normal(
                mean=means[i],
                cov=covariances[i],
                size=self.num_samples
            )
        self.trajectories = samples
        return samples

    def compute_weights(self, costs):
        """
        w_i = exp(-cost_i / lambda)
        """
        weights = np.exp(-costs / self.lambda_)
        weights /= np.sum(weights)
        self.weights = weights
        return weights

    def integrate_path(self):
        """
        compute weighted trajectory 
        """
        if self.trajectories is None or self.weights is None:
            raise ValueError("Trajectories or weights not computed.")
        self.best_traj = np.tensordot(self.weights, self.trajectories, axes=(0, 0))
        return self.best_traj

    def cost_func(self, trajs: np.ndarray, agent: Agent) -> np.ndarray:
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
        target = agent.get_target()[:2, 0]
        r = agent.r
        num_samples, horizon, _ = trajs.shape

        pos = trajs[:, :, :2]   # (N, T, 2)
        dist_goal = np.linalg.norm(pos - target[None, None, :], axis=-1)  # (N, T)
        c_goal = self.w_goal * dist_goal

        if omap is not None:
            c_obs = np.zeros_like(dist_goal)
            for k in range(num_samples):
                c_obs[k] = np.array([omap.cost(p) for p in pos[k]])
            c_obs *= self.w_obs
        else:
            c_obs = 0

        c_agent = np.zeros_like(dist_goal)
        if env is not None:
            for other in env._agents:
                if other is agent:
                    continue
                other_pos = np.array([v.mean[:2, 0] for v in other._vnodes])  # (T, 2)
                for k in range(num_samples):
                    # agent trajectory - other trajectory
                    diff = pos[k] - other_pos[None, :, :]
                    dist = np.linalg.norm(diff, axis=-1)
                    # 가까우면 큰 penalty (safe_dist 이하에서 급격히 증가)
                    safe_dist = r + other.r
                    penalty = np.exp(-0.5 * (dist / safe_dist)**2)
                    c_agent[k] += self.w_agent * penalty

        total_costs = np.sum(c_goal + c_obs + c_agent, axis=1)

        return total_costs

