import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict
import math
from scipy.stats import linregress

class EXP3E:
    def __init__(self, n_arms: int = 2, n_rounds: int = 1000, alpha: float = 0.1, delta: float = 0.05):
        """
        Initialize the EXP3E algorithm
        
        Args:
            n_arms: Number of arms (default=2 as per paper)
            n_rounds: Total number of rounds
            alpha: Exploration parameter
            delta: Confidence parameter
        """
        self.n_arms = n_arms
        self.n_rounds = n_rounds
        self.alpha = alpha
        self.delta = delta
        self.C = 4 * (math.e**2 + 2)**2 * (math.log(2/delta))**2 # initializes C = 4(e^2 + 2)^2(log(2/δ))^2
        
        # Initialize estimates and active arms
        self.R_hat = np.zeros(n_arms) # intializes R_hat(a) = 0 for all arms
        self.active_arms = list(range(n_arms)) # initializes A_1 = {1, 2}
        self.t = 1 # time steps
        
        # Store history for analysis
        self.history = {
            'chosen_arms': [],
            'rewards': [],
            'probabilities': [],
            'cumulative_regret': [],
            'ate_estimates': []
        }
        
    def _update_epsilon(self) -> float:
        """Calculate epsilon for the current round"""
        return 1 / math.sqrt(self.C * self.t) # calculates epsilon_t = 1/sqrt(Ct)
    
    def _update_alpha_t(self) -> float:
        """Calculate alpha for the current round"""
        return self.alpha / (2 * self.t) #  calculates alpha_t = alpha/2t
    
    def _get_probabilities(self) -> np.ndarray:
        """Calculate probabilities for arm selection"""
        probs = np.zeros(self.n_arms)
        
        if len(self.active_arms) == self.n_arms:  # Phase 1
            epsilon = self._update_epsilon()
            exp_rewards = np.exp(epsilon * self.R_hat)
            probs = exp_rewards / np.sum(exp_rewards)
        else:  # Phase 2
            alpha_t = self._update_alpha_t()
            # Distribute alpha_t equally among non-best arms
            non_best_prob = alpha_t / (self.n_arms - 1)
            probs.fill(non_best_prob)
            
            # Best arm gets remaining probability
            best_arm = np.argmax(self.R_hat)
            probs[best_arm] = 1 - alpha_t
        
        return probs
    
    def _update_active_arms(self):
        """Update the set of active arms based on reward estimates"""
        R_max = np.max(self.R_hat)
        threshold = 2 * math.sqrt(self.C / self.t)
        self.active_arms = [arm for arm in self.active_arms 
                          if R_max - self.R_hat[arm] <= threshold]
    
    def select_arm(self) -> tuple[int, np.ndarray]:
        """Select an arm based on current probabilities"""
        probs = self._get_probabilities()
        """
        If the arm is an active arm, the best arm (based on estimated cumulative reward) gets the majority 
        of the selection probability, meaning most of the probability mass is concentrated on the arm considered 
        optimal at time.
        For all other active arms (i.e., those not considered the best arm), the probabilities are 
        distributed equally using alpha_t to make sure other arms are explored even if less likely to be optimal
        """
        chosen_arm = np.random.choice(self.n_arms, p=probs) # uses the probabilities π_t(a) to select an arm
        return chosen_arm, probs
    
    def update(self, chosen_arm: int, reward: float, probs: np.ndarray):
        """Update estimates based on observed reward"""
        # Update reward estimate using importance sampling
        self.R_hat[chosen_arm] += reward / probs[chosen_arm]
        
        # Update active arms
        self._update_active_arms()
        
        # Store history
        self.history['chosen_arms'].append(chosen_arm)
        self.history['rewards'].append(reward)
        self.history['probabilities'].append(probs)
        
        # Calculate ATE estimate
        ate_estimate = (self.R_hat[0] - self.R_hat[1]) / self.t
        self.history['ate_estimates'].append(ate_estimate)
        
        # Increment time step
        self.t += 1
    
    def run_experiment(self, true_means: List[float]) -> Dict:
        """
        Run the complete experiment
        
        Args:
            true_means: List of true mean rewards for each arm
            
        Returns:
            Dictionary containing experimental results
        """
        optimal_arm = np.argmax(true_means)
        cumulative_regret = 0
        
        for _ in range(self.n_rounds):
            # Select arm
            chosen_arm, probs = self.select_arm()
            
            # Generate reward (as specified in the paper)
            noise = np.random.uniform(-1, 1)
            reward = true_means[chosen_arm] + noise
            
            # Update algorithm
            self.update(chosen_arm, reward, probs)
            
            # Calculate regret
            instant_regret = true_means[optimal_arm] - true_means[chosen_arm]
            cumulative_regret += instant_regret
            self.history['cumulative_regret'].append(cumulative_regret)
        
        return {
            'cumulative_regret': self.history['cumulative_regret'],
            'ate_estimates': self.history['ate_estimates'],
            'chosen_arms': self.history['chosen_arms'],
            'rewards': self.history['rewards']
        }

class TheoreticalBoundsVerification:
    def __init__(self, n_trials: int = 50):
        self.n_trials = n_trials
        
    def run_verification(self, alpha_values: List[float], n_values: List[int], true_means: List[float]):
        """
        Run verification experiments for different values of α and n
        """
        results = {}
        for alpha in alpha_values:
            results[alpha] = {}
            for n in n_values:
                trial_results = []
                for _ in range(self.n_trials):
                    exp3e = EXP3E(n_arms=2, n_rounds=n, alpha=alpha, delta=0.05)
                    result = exp3e.run_experiment(true_means)
                    trial_results.append({
                        'regret': result['cumulative_regret'][-1],
                        'ate_errors': [abs(ate - (true_means[0] - true_means[1])) 
                                     for ate in result['ate_estimates']]
                    })
                results[alpha][n] = self._aggregate_trials(trial_results)
        return results
    
    def _aggregate_trials(self, trial_results: List[Dict]) -> Dict:
        """Aggregate results across trials"""
        regrets = [r['regret'] for r in trial_results]
        ate_errors = np.mean([r['ate_errors'] for r in trial_results], axis=0)
        
        return {
            'mean_regret': np.mean(regrets),
            'std_regret': np.std(regrets),
            'mean_ate_error': ate_errors,
            'final_ate_error': ate_errors[-1]
        }
    
    def verify_bounds(self, results: Dict) -> Dict:
        """
        Verify if results match theoretical bounds
        Returns empirical exponents for comparison with theory
        """
        verification = {}
        
        for alpha in results.keys():
            # Get n values and corresponding results
            n_values = sorted(list(results[alpha].keys()))
            regrets = [results[alpha][n]['mean_regret'] for n in n_values]
            final_ate_errors = [results[alpha][n]['final_ate_error'] for n in n_values]
            
            # Fit log-log regression for regret
            log_n = np.log(n_values)
            log_regret = np.log(regrets)
            regret_slope, _, _, _, _ = linregress(log_n, log_regret)
            
            # log log exponent of the standard error of the ATE estimate
            log_ate_error = np.log(final_ate_errors)
            ate_slope, _, _, _, _ = linregress(log_n, log_ate_error)
            
            verification[alpha] = {
                'regret_exponent': regret_slope,
                'theoretical_regret_exponent': 1 - alpha,
                'ate_error_exponent': ate_slope,
                'theoretical_ate_error_exponent': -0.5 * (1 - alpha) # no solution can do better than a constant order in the worst case
            }
            
        return verification

def plot_verification_results(verification_results: Dict):
    """Plot comparison of empirical vs theoretical bounds (theoretical as according to the paper)"""
    alphas = sorted(list(verification_results.keys()))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot regret exponents
    empirical_regret = [verification_results[a]['regret_exponent'] for a in alphas]
    theoretical_regret = [verification_results[a]['theoretical_regret_exponent'] for a in alphas]
    
    ax1.plot(alphas, empirical_regret, 'bo-', label='Empirical')
    ax1.plot(alphas, theoretical_regret, 'r--', label='Theoretical')
    ax1.set_title('Regret Exponent vs α')
    ax1.set_xlabel('α')
    ax1.set_ylabel('Exponent')
    ax1.legend()
    ax1.grid(True)
    
    # Plot ATE error exponents
    empirical_ate = [verification_results[a]['ate_error_exponent'] for a in alphas]
    theoretical_ate = [verification_results[a]['theoretical_ate_error_exponent'] for a in alphas]
    
    ax2.plot(alphas, empirical_ate, 'bo-', label='Empirical')
    ax2.plot(alphas, theoretical_ate, 'r--', label='Theoretical')
    ax2.set_title('ATE Error Exponent vs α')
    ax2.set_xlabel('α')
    ax2.set_ylabel('Exponent')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    return fig

if __name__ == "__main__":
    # Parameters for verification
    alpha_values = [0.2, 0.4, 0.6, 0.8]
    n_values = [1000, 2000, 4000, 8000]
    true_means = [0.8, 0.2]
    
    # Run verification
    verifier = TheoreticalBoundsVerification(n_trials=20)
    results = verifier.run_verification(alpha_values, n_values, true_means)
    verification = verifier.verify_bounds(results)
    
    # Print results
    print("\nVerification Results:")
    for alpha, metrics in verification.items():
        print(f"\nα = {alpha}:")
        print(f"Regret: O(n^{metrics['regret_exponent']:.3f}) vs Theoretical O(n^{metrics['theoretical_regret_exponent']:.3f})")
        print(f"ATE Error: O(n^{metrics['ate_error_exponent']:.3f}) vs Theoretical O(n^{metrics['theoretical_ate_error_exponent']:.3f})")
    
    # Plot results
    fig = plot_verification_results(verification)
    plt.show()