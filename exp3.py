import numpy as np

class EXP3:
    """
    Implementation of the EXP3 (Exponential-weight algorithm for Exploration and Exploitation) algorithm
    for the adversarial multi-armed bandit problem.
    
    Parameters:
    -----------
    n_arms : int
        Number of arms/actions
    gamma : float
        Exploration parameter (0 <= gamma <= 1)
    eta : float, optional
        Learning rate. If None, it's set to sqrt(ln(n_arms)/(n_arms*T)) where T is horizon
    horizon : int, optional
        Number of rounds. Used to set eta if not provided explicitly
    """
    
    def __init__(self, n_arms, gamma, eta=None, horizon=None):
        self.n_arms = n_arms
        self.weights = np.ones(n_arms)
        self.gamma = gamma
        
        # Set learning rate
        if eta is None:
            if horizon is None:
                raise ValueError("Either eta or horizon must be specified")
            self.eta = np.sqrt(np.log(n_arms) / (n_arms * horizon))
        else:
            self.eta = eta
            
        # Initialize cumulative rewards and counts
        self.cum_rewards = np.zeros(n_arms)
        self.pulls = np.zeros(n_arms)
        self.t = 0
        
    def select_arm(self):
        """
        Select an arm according to the EXP3 algorithm
        
        Returns:
        int : The selected arm index
        """
        # Compute probabilities with mixed distribution
        probs = self._get_probs()
        
        # Select arm
        arm = np.random.choice(self.n_arms, p=probs)
        
        return arm
    
    def _get_probs(self):
        """Calculate the probability distribution over arms"""
        sum_weights = np.sum(self.weights)
        p_uniform = np.ones(self.n_arms) / self.n_arms
        p_weights = self.weights / sum_weights
        
        # Mix uniform exploration with weight-based exploitation
        probs = (1 - self.gamma) * p_weights + self.gamma * p_uniform
        
        return probs
    
    def update(self, arm, reward):
        """
        Update the algorithm parameters based on the observed reward
        
        Parameters:
        arm : int
            The arm that was pulled
        reward : float
            The observed reward (typically in [0,1])
        """
        self.t += 1
        self.pulls[arm] += 1
        self.cum_rewards[arm] += reward
        
        # Get the current probabilities
        probs = self._get_probs()
        
        # Compute importance-weighted estimated reward
        # This is a key part of EXP3: dividing by probability to create unbiased estimator
        estimated_reward = np.zeros(self.n_arms)
        estimated_reward[arm] = reward / probs[arm]
        
        # Update the weights using multiplicative weight update
        self.weights *= np.exp(self.eta * estimated_reward)
        
    def reset(self):
        """Reset the algorithm"""
        self.weights = np.ones(self.n_arms)
        self.cum_rewards = np.zeros(self.n_arms)
        self.pulls = np.zeros(self.n_arms)
        self.t = 0

# Example usage
if __name__ == "__main__":
    # Parameters
    n_arms = 5
    gamma = 0.1
    horizon = 1000
    
    # Initialize EXP3
    exp3 = EXP3(n_arms=n_arms, gamma=gamma, horizon=horizon)
    
    # Simulate rewards (normally you'd get these from the environment)
    # Here we use a simple environment where each arm has a fixed reward probability
    true_probs = [0.2, 0.3, 0.7, 0.1, 0.5]
    
    # Run for T rounds
    total_reward = 0
    rewards_per_round = []
    
    for t in range(horizon):
        # Select arm
        arm = exp3.select_arm()
        
        # Get reward (in a real scenario, this would come from the environment)
        reward = 1 if np.random.random() < true_probs[arm] else 0
        
        # Update algorithm
        exp3.update(arm, reward)
        
        # Track performance
        total_reward += reward
        rewards_per_round.append(reward)
        
        # Print occasional status
        if (t+1) % 100 == 0:
            print(f"Round {t+1}, Average Reward: {total_reward/(t+1)}")
    
    # Print final arm counts and average reward
    print(f"Arm pulls: {exp3.pulls}")
    print(f"Final average reward: {total_reward/horizon}")