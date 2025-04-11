
import numpy as np
import matplotlib.pyplot as plt

main
def realizable_estimator(context, reward, M, d, arm_index, policy_fn):
    """
    Implements Matrix Geometric Resampling (MGR) based realizable estimator from the Neu 2020 paper

    Parameters:
    - context: np.ndarray, the context vector X_t
    - reward: float, observed reward r_t
    - M: int, number of MGR iterations
    - d: int, context dimension
    - arm_index: int, selected arm
    - policy_fn: function that returns arm given a context

    Returns:
    - theta_hat: estimated reward vector for arm_index
    """

    beta = 1 / (2 * np.linalg.norm(context)**2 + 1e-8)
    I = np.eye(d)
    A_k = I
    Sigma_hat_plus = beta * I

    for _ in range(M):
        x_k = np.random.randn(d)
        x_k /= np.linalg.norm(x_k) + 1e-8
        a_k = policy_fn(x_k)
        if a_k == arm_index:
            B_k = np.outer(x_k, x_k)
            A_k = A_k @ (I - beta * B_k)
            Sigma_hat_plus += beta * A_k

    return Sigma_hat_plus @ context * reward # estimated reward vector theta_hat



main
class LinEXP3E:
    """
    Implementation of the modified RealLinEXP3 (Realizable Linear Exponential-weight algorithm for Exploration and Exploitation) algorithm
    to minimize cumulative regret while maintaining accure ATE estimation using forced expoloration.

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
    alpha: float
        Balance between regret minimization and ATE estimation accuracy (0 <= alpha <= 1)
    """

main
    def __init__(self, n_arms, gamma, context_dimension, eta=None, n_rounds=None, true_theta=None):
        self.n_arms = n_arms
        self.gamma = gamma
        self.context_dimension = context_dimension
        self.true_theta = true_theta  # for computing ATE estimation error


    def __init__(self, n_arms, gamma, context_dimension, eta=None, n_rounds=None):
        self.n_arms = n_arms # Number of arms (K)
        self.gamma = gamma # Exploration parameter
        self.context_dimension = context_dimension

        # Set learning rate if not provided
main
        if eta is None:
            if n_rounds is None:
                raise ValueError("If eta is not provided, n_rounds must be specified")
            self.eta = np.sqrt(np.log(n_arms) / (n_arms * n_rounds))
        else:
            self.eta = eta

main
        self.reward_estimator = np.zeros((n_arms, context_dimension))
        self.ate_estimator = np.zeros((n_arms, context_dimension))
        
        self.ate_update_counts = np.zeros(n_arms)  # counts of updates for each arm
        # for running average of reward estimators per arm

    def get_weights(self, context):
        scores = np.array([
            np.dot(context, self.reward_estimator[a]) for a in range(self.n_arms)
        ])
        weights = np.exp(self.eta * scores)
        return weights

    def draw_action(self, context, t):
        epsilon = 1.0 / (t + 1)  # decays over time
        if np.random.rand() < epsilon:
            return np.random.choice(self.n_arms)  # explore
        else:
            scores = np.array([
                np.dot(self.reward_estimator[a], context) for a in range(self.n_arms)
            ])
            return np.argmax(scores)

    def update_real_lin_exp3_reward_estimator(self, context, M, arm, reward, t=1):
        """
        Update the reward and ATE estimators using the realizable estimator.
        """
        # use realizable estimator from REALLINEXP3 paper
        reward_estimator = realizable_estimator(
            context=context,
            reward=reward,
            M=M,
            d=self.context_dimension,
            arm_index=arm,
            policy_fn=lambda x: self.draw_action(x, t)
        )

        # fast update for reward (EXP3 policy)
        fast_alpha = 0.2
        self.reward_estimator[arm] = (
            (1 - fast_alpha) * self.reward_estimator[arm] + fast_alpha * reward_estimator
        )

        # slow update for ATE tracking
        self.ate_update_counts[arm] += 1
        count = self.ate_update_counts[arm]
        self.ate_estimator[arm] = (
            (self.ate_estimator[arm] * (count - 1) + reward_estimator) / count
        )

        return reward_estimator


    # def update_real_lin_exp3_reward_estimator(self, context, M, arm, reward):
    #     d = self.context_dimension
    #     beta = 1 / (2 * np.linalg.norm(context)**2)
    #     I = np.eye(d)
    #     A_k = I
    #     Sigma_hat_plus = beta * I

    #     for _ in range(M):
    #         x_k = np.random.randn(d)
    #         x_k /= np.linalg.norm(x_k) # normalize to reduce variance
    #         a_k = self.draw_action(x_k, _)

    #         if a_k == arm:
    #             B_k = np.outer(x_k, x_k)
    #             A_k = A_k @ (I - beta * B_k)
    #             Sigma_hat_plus += beta * A_k


    #     reward_estimator = Sigma_hat_plus @ context * reward # realizable estimator

    #     # update reward estimator (EXP3) -- more responsive
    #     fast_alpha = 0.2 or 0.3
    #     self.reward_estimator[arm] = (
    #         (1 - fast_alpha) * self.reward_estimator[arm] + fast_alpha * reward_estimator
    #     )

    #     # update ATE estimator -- slow moving average
    #     self.ate_update_counts[arm] += 1
    #     count = self.ate_update_counts[arm]
    #     self.ate_estimator[arm] = (
    #         (self.ate_estimator[arm] * (count - 1) + reward_estimator) / count
        
        # self.reward_estimator[arm] += reward_estimator # too aggressive?
        # self.reward_estimator[arm] = ( self.reward_estimator[arm] * t + reward_estimator) / (t + 1)
        # self.arm_update_counts[arm] += 1 # ATE error issue (local minimum?)
        # count = self.arm_update_counts[arm]
        # self.reward_estimator[arm] = (self.reward_estimator[arm] * (count - 1) + reward_estimator) / count

        # use a weighted moving average to give importance to more recent updates
        # alpha = 0.01

        # adapting alpha over time
        # alpha = 0.1

        # self.arm_update_counts[arm] += 1
        # alpha = 1.0/self.arm_update_counts[arm] # decaying alpha
        # self.reward_estimator[arm] = (1 - alpha) * self.reward_estimator[arm] + alpha * reward_estimator

        return reward_estimator

    def get_ate_errors(self):
        """
        Compute ATE estimation error (L2 norm) for each arm.

        Was running into issue where learning dominates in early rounds, so ATE error decreases quickly and variance is low. Then, in later rounds, noise and overfitting begin to affect estimates and ATE error increases.
        This is likely due to the updates not being weighted properly, thus allowing noisy and non-optimal updates to accumpulate. Need to fix weighting.

        Once I fixed that (by adapting alpha over time), it meant that of course reward optimization was much worse because the agent is too cautious and failing to adapt its policy to maximize reward. Figured out that EXP3-style probabilistic sampling was the issue, because the agent was exploring too much even when the best action was clear. Going greedy let the agent exploit what it had learned, and increased rewards significantly
        """
    
        if self.true_theta is None:
            return None
        errors = np.linalg.norm(self.ate_estimator - self.true_theta, axis=1)
        # errors = np.linalg.norm(self.reward_estimator - self.true_theta, axis=1)
        return errors
    


        # Set reward vector
        self.reward_estimator = np.zeros((n_arms, context_dimension)) # Shape (K, d)

        # Set policies
        probs = (1-self.gamma) * weights / np.sum(weights) + (self.gamma / self.n_arms)


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
        
        self.last_context = context
        self.last_probs = probs
      
        arm = np.random.choice(self.n_arms, p=probs)

        # If a is suboptimal:
        probs[a] = (1-self.alpha) * weights[a] / np.sum(weights[a]) + (self.alpha / self.n_arms)
        # Else: 
        probs[a] = (1-self.gamma) * weights[a] / np.sum(weights[a]) + (self.gamma / self.n_arms)

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

main
if __name__ == "__main__":
    # Parameters
    n_arms = 5
    context_dim = 10
main
    gamma = 0.01
=======
    gamma = 0.1
main
    horizon = 1000
    M = 10  # MGR samples

    # True unknown reward parameters (one per arm)
    true_theta = np.random.randn(n_arms, context_dim)

    # Initialize LinEXP3E agent
main
    # Initialize LinEXP3E agent
    agent = LinEXP3E(
        n_arms=n_arms,
        gamma=gamma,
        context_dimension=context_dim,
        n_rounds=horizon,
        true_theta=true_theta  # <-- this must be here!
    )

    agent = LinEXP3E(n_arms=n_arms, gamma=gamma, context_dimension=context_dim, n_rounds=horizon)
main

    # Tracking performance
    total_reward = 0
    total_regret = 0
    rewards_per_round = []
    regret_per_round = []
main
    ate_errors_per_round = []

    for t in range(horizon):
        context = np.random.randn(context_dim)
        arm = agent.draw_action(context, t)


        estimated_rewards = [np.dot(agent.reward_estimator[a], context) for a in range(n_arms)]
        print(f"Round {t}, Estimated Rewards: {estimated_rewards}")
        predicted_best_arm = np.argmax(estimated_rewards)
        print(f"Chosen Arm: {arm}, Predicted Best Arm: {predicted_best_arm}")

        reward = np.dot(true_theta[arm], context) + np.random.normal(0, 0.01) # changed from 0.1 to 0.01 to reduce noise temporarily
        optimal_reward = np.max([np.dot(theta, context) for theta in true_theta])
        regret = optimal_reward - reward

        agent.update_real_lin_exp3_reward_estimator(context, M=M, arm=arm, reward=reward, t=t)
        # agent.update_real_lin_exp3_reward_estimator(context, M=M, arm=arm, reward=reward)


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
main

        # Track performance
        total_reward += reward
        total_regret += regret
        rewards_per_round.append(total_reward / (t + 1))
        regret_per_round.append(total_regret)

main
        # ATE estimation error
        ate_errors = agent.get_ate_errors()
        if ate_errors is not None:
            mean_ate_error = np.mean(ate_errors)
            ate_errors_per_round.append(mean_ate_error)
            
        print(f"Round {t}, Mean ATE Error: {mean_ate_error}")

main
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
main

    # ATE Estimation Error
    plt.figure(figsize=(6, 4))
    plt.plot(ate_errors_per_round)
    plt.title("ATE Estimation Error Over Time")
    plt.xlabel("Rounds")
    plt.ylabel("Mean L2 Error")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

 main
