# -*- coding: utf-8 -*-
import sys
import math
import numpy as np
import matplotlib.pyplot as plt

class LinEXP3E:
    """
    Implementation of the RealLinEXP3 (Realizable Linear Exponential-weight algorithm for Exploration and Exploitation) algorithm
    for the adversarial multi-armed bandit problem with additions for ATE (Average Treatment Effect) estimation.

    Parameters:
    -----------
    n_arms : int
        Number of arms/actions
    gamma : float
        Exploration parameter (0 <= gamma <= 1)
    eta : float, optional
        Learning rate. If None, it's set to sqrt(ln(n_arms)/(n_arms*T)) where T is n_rounds
    n_rounds : int, optional
        Number of rounds. Used to set eta if not provided explicitly
    context_dimension : int
        Context dimension
    """

    def __init__(self, n_arms, gamma, context_dimension, eta=None, n_rounds=None):
        self.n_arms = n_arms # Number of arms (K)
        self.gamma = gamma # Exploration parameter
        self.context_dimension = context_dimension
        self.reward_estimator_IPW = {a: [] for a in range(self.n_arms)} # list of individual IPW reward values per round
        self.ATE_matrix = {
            a: {b: [] for b in range(self.n_arms) if b != a}
            for a in range(self.n_arms)
        }

        self.R_hat = np.zeros(self.n_arms) # running estimate of cumulative rewards per arm

        # Set learning rate if not provided
        if eta is None:
            if n_rounds is None:
                raise ValueError("If eta is not provided, n_rounds must be specified")
            if n_rounds <= 0:
                raise ValueError("n_rounds must be greater than zero")
            self.eta = np.sqrt(np.log(n_arms) / (n_arms * n_rounds))
        else:
            self.eta = eta

        # Set reward vector
        self.reward_estimator = np.zeros((n_arms, context_dimension)) # Shape (K, d)


    def get_weights(self, context):
        """
        Observe the current context vector (X_t) and for all a, set weights
        """
        scores = np.array([
            np.dot(context, self.reward_estimator[a]) for a in range(self.n_arms)
        ])

        weights = np.exp(self.eta * scores)

        return weights
    
    def get_action_probs(self, context):
        """
        Return softmax probabilities over contextual scores since get_weights gives exponentiated scores.
        Returns all probabilities for each arm in a single array.
        Uses the same policy given in LinEXP3.
        """

        weights = self.get_weights(context)
        probs = (1-self.gamma) * weights / np.sum(weights) + (self.gamma / self.n_arms)
        return probs
    
    def draw_action(self, context):
        """
        Draw an arm (A_t) using the policy (π) for context (x_t)
        """
        probs = self.get_action_probs(context)
        arm = int(np.random.choice(self.n_arms, p=probs))
        return arm

    def get_action_probability(self, context, arm):
        """
        Wrapper function to return π(a | x_t) for a given context and arm,
        using the same policy as draw_action (EXP3-style with gamma smoothing).
        """
        return self.get_action_probs(context)[arm]
    
    def matrix_geometric_resampling(self, context, M):
        """
        Matrix Geometric Resampling (MGR) for estimating the reward function.
        This is a placeholder function and should be implemented based on the
        specific MGR algorithm used in the paper.

        Parameters
        ----------
        context : np.ndarray
            Context vector observed at current round.
        M : int
            Number of resampling iterations.
        """
        d = self.context_dimension
        Beta = 1 / (2 * np.linalg.norm(context)**2)
        I = np.eye(d)
        A_k = I
        Sigma_inv_est = Beta * I

        for k in range(M):
            # draw context matrix from D
            x_k = np.random.randn(d)

            # select action from probs using context matrix
            a_k = self.draw_action(x_k)

            if a_k == arm:
                # compute B_k,a
                B_k = np.outer(x_k, x_k)

                # compute A_k,a
                A_k = A_k @ (I - Beta * B_k)

                # return MGR estimator (context-aware covariance estimate)
                Sigma_inv_est += Beta * A_k

        return Sigma_inv_est

    def update_real_lin_exp3_reward_estimator(self,context,M,arm,reward):
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
        """
        inv_cov_matrix_est = self.matrix_geometric_resampling(self, context, M)


        # Calculate reward estimator using context just learned
        reward_estimator = inv_cov_matrix_est @ context * reward
        self.reward_estimator[arm] += reward_estimator

        # Get probability for given arm given context
        prob = self.get_action_probability(context, arm)
       
        # do IPW to adjust reward  based on how likely action was selected since we don't see reward of both arms at once
        inv_propensity = 1.0 / max(prob, 1e-6) # 1/pi[arm]
        ipw_reward = reward_estimator * inv_propensity # do IPW with contextual reward estimate
        
        # keep a running list to check convergence
        self.reward_estimator_IPW[arm].append(ipw_reward) # list of individual IPW reward values per round, not contextual

        for other_arm in range(self.n_arms): # loop through all other arms
            if other_arm == arm: # other than just-pulled one
                continue
            if self.reward_estimator_IPW[other_arm]: # only if we have data for comparison
                other_mean = np.mean(self.reward_estimator_IPW[other_arm]) # pull mean IPW reward for other arm
                ate = ipw_reward - other_mean # compare it to reward of current arm
                self.ATE_matrix[arm][other_arm].append(ate) # update ATE data structure accordingly

        # return reward estimator
        return reward_estimator # define another self.reward_estimator_IPW, compute all pairwise differences.
        """
        Vector of length k choose 2, each facilitates hypothesis testing that arm A versus arm A^ is 
        better or worse for all pairs. If ATE converges to 0, reject null hypothesis.
        attribute for IPW'd estimate (*1/pi[arm])
        maintain data structure for ATE for all arms, expect each entry to converge to a constant.
        """


def plot_ate_convergence(agent):
    n_arms = agent.n_arms
    arm_pairs = [(a, b) for a in range(n_arms) for b in range(n_arms) if a != b]

    n_plots = len(arm_pairs)
    n_cols = 3
    n_rows = math.ceil(n_plots / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 1.25 * n_rows))
    axes = axes.flatten() # in case it is 2-D array

    global_min = -1
    global_max = 1

    for idx, (a, b) in enumerate(arm_pairs):
        ates = agent.ATE_matrix[a][b]
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

    # hide any leftover empty subplots
    for j in range(idx + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Parameters
    n_arms = 5
    context_dim = 10
    gamma = 0.1
    horizon = 1000
    M = 10  # MGR samples

    # True unknown reward parameters (one per arm)
    true_theta = np.random.randn(n_arms, context_dim)

    # Initialize LinEXP3 agent
    agent = LinEXP3E(n_arms=n_arms, gamma=gamma, context_dimension=context_dim, n_rounds=horizon)

    # Tracking performance
    total_reward = 0
    total_regret = 0
    rewards_per_round = []
    regret_per_round = []

    for t in range(horizon):
        # Generate random context (from D ~ N(0, I))
        
        context = np.random.randn(context_dim)

        # Choose action
        arm = agent.draw_action(context)
        # print(arm)

        # Compute reward from true linear model + noise
        reward = np.dot(true_theta[arm], context) + np.random.normal(0, 0.1)  # noise std = 0.1

        # Compute optimal reward for regret tracking
        optimal_reward = np.max([np.dot(theta, context) for theta in true_theta])
        regret = optimal_reward - reward

        # Update LinEXP3
        agent.update_real_lin_exp3_reward_estimator(context, M=M, arm=arm, reward=reward)

        # Track performance
        total_reward += reward
        total_regret += regret
        
        rewards_per_round.append(total_reward / (t + 1))
        regret_per_round.append(total_regret)
        avg_regret_per_round = [r / (t+1) for t, r in enumerate(regret_per_round)]

    plot_ate_convergence(agent)

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

