# -*- coding: utf-8 -*-
import math
import numpy as np
import matplotlib.pyplot as plt

class LinEXP3E:
    """
    Implementation of the RealLinEXP3 algorithm for adversarial contextual bandits 
    with Average Treatment Effect (ATE) estimation.
    
    Parameters:
    -----------
    n_arms : int
        Number of arms/actions (K > 0)
    gamma : float
        Exploration parameter in [0, 1]
    context_dimension : int
        Dimension of the context vectors (d > 0)
    eta : float, optional
        Learning rate. If None, computed as sqrt(ln(K)/(KT))
    n_rounds : int, optional
        Number of rounds (T). Required if eta is None
    """
    def __init__(self, n_arms: int, gamma: float, context_dimension: int, 
                 eta: float = None, n_rounds: int = None) -> None:
        # Validate inputs
        if n_arms <= 0 or context_dimension <= 0:
            raise ValueError("n_arms and context_dimension must be positive")
        if not 0 <= gamma <= 1:
            raise ValueError("gamma must be in [0, 1]")
            
        self.n_arms = n_arms
        self.gamma = gamma
        self.context_dimension = context_dimension
        
        # Initialize estimators and statistics
        self.reward_estimator = np.zeros((n_arms, context_dimension))  # θ_t for each arm
        self.reward_estimator_IPW = {a: [] for a in range(n_arms)}  # IPW rewards history
        self.cumulative_rewards = np.zeros(n_arms)  # R_hat in the paper
        
        # ATE tracking matrix: arm a vs arm b
        self.ATE_matrix = {
            a: {b: [] for b in range(n_arms) if b != a}
            for a in range(n_arms)
        }

        # Set learning rate
        if eta is None:
            if n_rounds is None or n_rounds <= 0:
                raise ValueError("n_rounds must be positive when eta is None")
            self.eta = np.sqrt(np.log(n_arms) / (n_arms * n_rounds))
        else:
            if eta <= 0:
                raise ValueError("eta must be positive")
            self.eta = eta

    def get_weights(self, context: np.ndarray) -> np.ndarray:
        """Calculate arm weights based on current context."""
        scores = np.array([
            np.dot(context, self.reward_estimator[a]) for a in range(self.n_arms)
        ])

        weights = np.exp(self.eta * scores)

        return weights

    def get_action_probs(self, context: np.ndarray) -> np.ndarray:
        """
        Return softmax probabilities over contextual scores since get_weights gives exponentiated scores.
        Returns all probabilities for each arm in a single array.
        Uses the same policy given in LinEXP3.
        """
        weights = self.get_weights(context)
        probs = (1 - self.gamma) * weights / np.sum(weights) + (self.gamma / self.n_arms)
        return probs

    def draw_action(self, context: np.ndarray) -> int:
        """
        Draw an arm (A_t) using the policy (π) for context (x_t)
        """
        probs = self.get_action_probs(context)
        arm = int(np.random.choice(self.n_arms, p=probs))
        return arm

    def get_action_probability(self, context: np.ndarray, arm: int) -> float:
        """
        Wrapper function to return π(a | x_t) for a given context and arm,
        using the same policy as draw_action (EXP3-style with gamma smoothing).
        """
        return self.get_action_probs(context)[arm]

    def matrix_geometric_resampling(self, context: np.ndarray, M: int, arm: int) -> np.ndarray:
        """
        Perform Matrix Geometric Resampling to estimate inverse covariance matrix.
        
        Args:
            context: Current context vector x_t
            M: Number of resampling iterations
            arm: Target arm for estimation
            
        Returns:
            Estimated inverse covariance matrix
        """
        d = self.context_dimension
        context_norm = np.linalg.norm(context)
        
        # Avoid numerical instability
        epsilon = 1e-10
        beta = 1.0 / (2 * max(context_norm**2, epsilon))
        
        identity = np.eye(d)
        A_current = identity.copy()
        Sigma_inv_est = beta * identity

        for _ in range(M):
            # Sample random context and action
            context_sample = np.random.randn(d)
            action = self.draw_action(context_sample)
            
            if action == arm:
                outer_product = np.outer(context_sample, context_sample)
                A_current = A_current @ (identity - beta * outer_product)
                Sigma_inv_est += beta * A_current

        return Sigma_inv_est

    def update_real_lin_exp3_reward_estimator(self, context: np.ndarray, M: int, arm: int, reward: float) -> np.ndarray:
        """
        Return reward estimator based on current context and arm.

        Parameters
        ----------
        context : np.ndarray
            Context vector observed at current round.
        M : int
            Number of resampling iterations.
        arm : int
            Action selected by the agent.
        reward : float
            Observed reward for the selected action.

        Returns
        -------
        np.ndarray
            Updated reward estimator
        """
        inv_cov_matrix_est = self.matrix_geometric_resampling(context, M, arm)

        # Update reward estimator using inverse covariance matrix estimate
        reward_estimator = inv_cov_matrix_est @ context * reward
        self.reward_estimator[arm] += reward_estimator

        # Calculate inverse propensity weight (IPW)
        prob = self.get_action_probability(context, arm)
        inv_propensity = 1.0 / max(prob, 1e-6)  # Clipped IPW to avoid division by zero

        # Calculate IPW-adjusted reward estimate 
        ipw_reward = float(np.mean(reward_estimator * inv_propensity))  # Convert to scalar
        self.reward_estimator_IPW[arm].append(ipw_reward)

        # Update ATE estimates for all pairs
        curr_arm_reward = ipw_reward
        for other_arm in range(self.n_arms):
            if other_arm != arm and self.reward_estimator_IPW[other_arm]:
                other_arm_mean = float(np.mean(self.reward_estimator_IPW[other_arm]))
                ATE = curr_arm_reward - other_arm_mean
                self.ATE_matrix[arm][other_arm].append(ATE)

        return reward_estimator


def plot_ate_convergence(agent: LinEXP3E) -> None:
    """
    Visualize ATE convergence for all arm pairs over time.
    Creates a grid of subplots showing running averages of ATE estimates.
    """
    n_arms = agent.n_arms
    arm_pairs = [(a, b) for a in range(n_arms) for b in range(n_arms) if a != b]

    n_plots = len(arm_pairs)
    n_cols = 3
    n_rows = math.ceil(n_plots / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 1.25 * n_rows))
    axes = axes.flatten()  # Ensure axes is always flat array

    global_min, global_max = -1, 1

    for idx, (a, b) in enumerate(arm_pairs):
        ates = np.array(agent.ATE_matrix[a][b], dtype=float)
        if len(ates) > 0:
            running_avg = [np.mean(ates[:i+1]) for i in range(len(ates))]
            axes[idx].plot(running_avg)
            axes[idx].set_ylim(global_min, global_max)
            axes[idx].set_title(f"ATE({a} vs {b})")
            axes[idx].set_xlabel("Rounds")
            axes[idx].set_ylabel("Running ATE")
        else:
            axes[idx].text(0.5, 0.5, 'No Data', ha='center')
            axes[idx].set_xticks([])
            axes[idx].set_yticks([])

    # Hide empty subplots
    for j in range(idx + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Simulation parameters
    n_arms = 5  # Number of treatment arms
    context_dim = 10  # Feature dimension
    gamma = 0.1  # Exploration factor
    horizon = 1000  # Time horizon T
    M = 10  # MGR iterations
    noise_std = 0.1  # Standard deviation of reward noise

    # True unknown reward parameters (one per arm)
    true_theta = np.random.randn(n_arms, context_dim)

    # Initialize LinEXP3 agent
    agent = LinEXP3E(n_arms=n_arms, gamma=gamma, context_dimension=context_dim, n_rounds=horizon)

    # Tracking performance
    total_reward = 0
    total_regret = 0
    rewards_per_round = []
    regret_per_round = []

    # Run simulation
    for t in range(horizon):
        context = np.random.randn(context_dim)
        arm = agent.draw_action(context)

        # Compute true reward and noise
        reward = np.dot(true_theta[arm], context) + np.random.normal(0, noise_std)
        optimal_reward = np.max([np.dot(theta, context) for theta in true_theta])
        regret = optimal_reward - reward

        # Update agent and track metrics
        agent.update_real_lin_exp3_reward_estimator(context, M=M, arm=arm, reward=reward)
        total_reward += reward
        total_regret += regret
        rewards_per_round.append(total_reward / (t + 1))
        regret_per_round.append(total_regret)

    # Plot results
    plot_ate_convergence(agent)

    plt.figure(figsize=(12, 4))

    # Average reward
    plt.subplot(1, 2, 1)
    plt.plot(rewards_per_round)
    plt.title("Average Reward Over Time")
    plt.xlabel("Rounds")
    plt.ylabel("Average Reward")
    plt.grid(True)

    # Cumulative regret
    plt.subplot(1, 2, 2)
    plt.plot(regret_per_round, color='red')
    plt.title("Cumulative Regret Over Time")
    plt.xlabel("Rounds")
    plt.ylabel("Cumulative Regret")
    plt.grid(True)

    plt.tight_layout()
    plt.show()

