# -*- coding: utf-8 -*-
"""LinEXP3.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1nivdqsS4TaT7lA_xIJ17BLqMwBNrRNoa
"""

import numpy as np
import matplotlib.pyplot as plt

class LinEXP3:
    """
    Implementation of the RealLinEXP3 (Realizable Linear Exponential-weight algorithm for Exploration and Exploitation) algorithm
    for the adversarial multi-armed bandit problem.

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

        # Set learning rate if not provided
        if eta is None:
            if n_rounds is None:
                raise ValueError("If eta is not provided, n_rounds must be specified")
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


    def draw_action(self, context):
        """
        Draw an arm (A_t) from the policy based on pi(a|X_t)
        """
        weights = self.get_weights(context)
        probs = (1-self.gamma) * weights / np.sum(weights) + (self.gamma / self.n_arms)
        self.last_context = context
        self.last_probs = probs

        arm = np.random.choice(self.n_arms, p=probs)

        return arm


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
        # Matrix Geometrix Resampling
        d = self.context_dimension
        beta = 1 / (2 * np.linalg.norm(context)**2)
        I = np.eye(d)
        A_k = I
        Sigma_hat_plus = beta * I

        for k in range(M):
            # draw context matrix from D
            x_k = np.random.randn(d)

            # select action from probs
            a_k = self.draw_action(x_k)


            if a_k == arm:
                # compute B_k,a
                B_k = np.outer(x_k, x_k)

                # compute A_k,a
                A_k = A_k @ (I - beta * B_k)

                # return MGR estimator
                Sigma_hat_plus += beta * A_k


        # calculate reward estimator
        reward_estimator = Sigma_hat_plus @ context * reward
        self.reward_estimator[arm] += reward_estimator

        # return reward estimator
        return reward_estimator

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
    agent = LinEXP3(n_arms=n_arms, gamma=gamma, context_dimension=context_dim, n_rounds=horizon)

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

    # Plot
    plt.figure(figsize=(12, 5))

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
