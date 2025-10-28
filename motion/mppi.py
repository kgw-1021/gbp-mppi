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

    def __init__(self, num_samples=128):
        self.num_samples = num_samples
        self.best_traj = None
        self.trajectories = None
        self.weights = None
        self.w_target = 1.0
        self.w_obs = 5.0

 

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

    def cost_func(self, trajs: np.ndarray) -> np.ndarray:
        """
        Compute trajectory cost for all sampled trajectories.

        Args:
            trajs (np.ndarray): shape (num_samples, horizon, state_dim)

        Returns:
            np.ndarray: total cost per trajectory (shape: (num_samples,))
        """
        # positions only (N, T, 2)
        pos = trajs[:, :, :2]

        # --- target attraction ---
        target = self.target_mean[:2, 0]  # (2,)
        diff = pos - target[np.newaxis, np.newaxis, :]
        c_t = np.linalg.norm(diff, axis=-1)  # (N, T)

        # --- obstacle avoidance ---
        if self.omap is not None:
            c_o = np.zeros_like(c_t)
            for k in range(self.num_samples):
                c_o[k] = np.array([self.omap.cost(p) for p in pos[k]])
        else:
            c_o = np.zeros_like(c_t)

        # --- total cost ---
        total_costs = np.sum(self.w_target * c_t + self.w_obs * c_o, axis=1)

        return total_costs


