import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import time
import pandas as pd
from tqdm import tqdm

class DABConverter:
    """
    Dual Active Bridge (DAB) Converter model with Triple Phase Shift (TPS) modulation
    """
    def __init__(self, V1=100, V2=50, Lk=15e-6, fs=50e3, n=2, Pnom=1000, dt_max=0.5):
        """
        Initialize the DAB converter parameters
        
        Args:
            V1: Input voltage (V)
            V2: Output voltage referred to primary (V)
            Lk: Leakage inductance (H)
            fs: Switching frequency (Hz)
            n: Transformer turns ratio
            Pnom: Nominal power (W)
            dt_max: Maximum phase shift (normalized to 0.5)
        """
        self.V1 = V1
        self.V2 = V2
        self.Lk = Lk
        self.fs = fs
        self.n = n
        self.Pnom = Pnom
        self.dt_max = dt_max
        self.Ts = 1/fs
        self.k = V2/V1
        self.Zbase = 8 * self.fs * self.Lk
        self.Pbase = (self.V1**2) / self.Zbase
        
    def calculate_power(self, D1, D2, D3):
        """
        Calculate normalized power for given phase shift values.
        
        Args:
            D1: Phase shift between bridges (primary to secondary)
            D2: Phase shift within primary bridge (duty cycle)
            D3: Phase shift within secondary bridge (duty cycle)
            
        Returns:
            normalized_power: Power normalized to base power
            p_actual: Actual power in Watts
        """
        # Constrain phase shifts to valid range
        D1 = np.clip(D1, -self.dt_max, self.dt_max)
        D2 = np.clip(D2, 0, self.dt_max)
        D3 = np.clip(D3, 0, self.dt_max)
        
        # Calculate power using TPS equations
        if D1 >= 0:
            if D1 <= D3 and D1 <= D2:
                # Mode 1
                p = D1 * (1 - D1) - (D2 - D1)**2/2 - (D3 - D1)**2/2
                
            elif D2 <= D1 and D2 <= D3:
                # Mode 2
                p = D2 * (1 - D2/2) + D1 * (D2 - D1) - (D3 - D1)**2/2
                
            elif D3 <= D1 and D3 <= D2:
                # Mode 3
                p = D3 * (1 - D3/2) + D1 * (D3 - D1) - (D2 - D1)**2/2
                
            else:
                # Mode 4
                p = D1 + D2 * (D3 - D2/2) + D3 * (D2 - D3/2)
        else:
            # Negative D1 - negate power
            D1_abs = abs(D1)
            if D1_abs <= D3 and D1_abs <= D2:
                # Mode 1
                p = D1_abs * (1 - D1_abs) - (D2 - D1_abs)**2/2 - (D3 - D1_abs)**2/2
                
            elif D2 <= D1_abs and D2 <= D3:
                # Mode 2
                p = D2 * (1 - D2/2) + D1_abs * (D2 - D1_abs) - (D3 - D1_abs)**2/2
                
            elif D3 <= D1_abs and D3 <= D2:
                # Mode 3
                p = D3 * (1 - D3/2) + D1_abs * (D3 - D1_abs) - (D2 - D1_abs)**2/2
                
            else:
                # Mode 4
                p = D1_abs + D2 * (D3 - D2/2) + D3 * (D2 - D3/2)
            
            # Negate power for negative D1
            p = -p
            
        # Calculate normalized power and actual power
        normalized_power = self.k * p
        p_actual = normalized_power * self.Pbase
        
        return normalized_power, p_actual
    
    def calculate_rms_current(self, D1, D2, D3):
        """
        Calculate RMS current for TPS modulation.
        
        Args:
            D1: Phase shift between bridges (primary to secondary)
            D2: Phase shift within primary bridge (duty cycle)
            D3: Phase shift within secondary bridge (duty cycle)
            
        Returns:
            i_rms_normalized: Normalized RMS current
            i_rms_actual: Actual RMS current in Amperes
        """
        # Simplified RMS current calculation
        # In a real implementation, this would contain the full equations for RMS current
        # For this example, using a simplified model based on power
        p_norm, _ = self.calculate_power(D1, D2, D3)
        
        # Simplified RMS calculation
        i_rms_squared = abs(p_norm) * (abs(D1) + D2 + D3) / 3
        i_rms_normalized = np.sqrt(i_rms_squared)
        
        # Convert to actual current
        i_base = self.V1 / self.Zbase
        i_rms_actual = i_rms_normalized * i_base
        
        return i_rms_normalized, i_rms_actual
    
    def calculate_reward(self, D1, D2, D3, target_power, weight_power=1.0, weight_current=0.5):
        """
        Calculate reward based on power error and RMS current.
        
        Args:
            D1, D2, D3: Phase shift values
            target_power: Target power in Watts
            weight_power: Weight for power error component
            weight_current: Weight for current minimization component
            
        Returns:
            reward: Combined reward value
        """
        # Calculate power and error
        _, actual_power = self.calculate_power(D1, D2, D3)
        power_error = abs(actual_power - target_power)
        
        # Calculate current
        _, i_rms = self.calculate_rms_current(D1, D2, D3)
        
        # Normalize errors
        power_error_norm = power_error / self.Pnom
        current_norm = i_rms / (self.V1 / self.Zbase)
        
        # Calculate reward components
        power_reward = np.exp(-5 * power_error_norm)  # Higher when power error is lower
        current_reward = np.exp(-2 * current_norm)    # Higher when current is lower
        
        # Combine rewards
        combined_reward = weight_power * power_reward + weight_current * current_reward
        
        return combined_reward

class QLearningAgent:
    """
    Q-Learning agent for optimizing DAB converter phase shifts
    """
    def __init__(self, state_space, action_space, learning_rate=0.1, discount_factor=0.9, exploration_rate=1.0, exploration_decay=0.995, min_exploration=0.01):
        """
        Initialize Q-Learning agent
        
        Args:
            state_space: List of [min_value, max_value, num_states] for each state dimension
            action_space: List of [min_value, max_value, num_actions] for each action dimension 
            learning_rate: Learning rate alpha
            discount_factor: Discount factor gamma
            exploration_rate: Initial exploration rate (epsilon)
            exploration_decay: Rate at which exploration decreases
            min_exploration: Minimum exploration rate
        """
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.min_exploration = min_exploration
        
        # Define state and action spaces
        self.state_space = state_space
        self.action_space = action_space
        
        # Number of states and actions for each dimension
        self.state_dims = [int(s[2]) for s in state_space]
        self.action_dims = [int(a[2]) for a in action_space]
        
        # Create Q-table
        self.q_table = np.zeros(self.state_dims + self.action_dims)
        
        # Initialize action values
        self.action_values = []
        for i, (min_val, max_val, num_actions) in enumerate(action_space):
            self.action_values.append(np.linspace(min_val, max_val, int(num_actions)))
            
        # Track best actions found so far
        self.best_state = None
        self.best_action = None
        self.best_reward = -np.inf
    
    def state_to_index(self, state):
        """Convert continuous state to discrete index"""
        indices = []
        for i, (s, (min_val, max_val, num_states)) in enumerate(zip(state, self.state_space)):
            # Clip state to be within bounds
            s_clipped = max(min_val, min(s, max_val))
            # Convert to index
            idx = int((s_clipped - min_val) / (max_val - min_val) * (num_states - 1))
            # Ensure index is within bounds
            idx = max(0, min(idx, int(num_states) - 1))
            indices.append(idx)
        return tuple(indices)
    
    def choose_action(self, state):
        """Choose action using epsilon-greedy policy"""
        if np.random.rand() < self.exploration_rate:
            # Exploration: choose random action
            action_indices = [np.random.randint(0, dim) for dim in self.action_dims]
            action = [self.action_values[i][idx] for i, idx in enumerate(action_indices)]
            return action, tuple(action_indices)
        else:
            # Exploitation: choose best action from Q-table
            state_idx = self.state_to_index(state)
            action_indices = np.unravel_index(np.argmax(self.q_table[state_idx]), self.action_dims)
            action = [self.action_values[i][idx] for i, idx in enumerate(action_indices)]
            return action, action_indices
    
    def update_q_value(self, state, action_indices, reward, next_state):
        """Update Q-value using Q-learning algorithm"""
        state_idx = self.state_to_index(state)
        next_state_idx = self.state_to_index(next_state)
        
        # Get current Q-value
        current_q = self.q_table[state_idx + action_indices]
        
        # Get max Q-value for next state
        max_next_q = np.max(self.q_table[next_state_idx])
        
        # Update Q-value
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)
        self.q_table[state_idx + action_indices] = new_q
        
        # Update best action if this is better
        if reward > self.best_reward:
            self.best_reward = reward
            self.best_state = state
            self.best_action = [self.action_values[i][idx] for i, idx in enumerate(action_indices)]
    
    def decay_exploration(self):
        """Decay exploration rate"""
        self.exploration_rate = max(self.min_exploration, self.exploration_rate * self.exploration_decay)

def run_optimization(target_power=500, num_episodes=1000, visualize=True):
    """
    Run the Q-learning optimization process for the DAB converter
    
    Args:
        target_power: Target power in Watts
        num_episodes: Number of training episodes
        visualize: Whether to visualize the results
        
    Returns:
        best_params: Best phase shift parameters found (D1, D2, D3)
        best_power: Actual power achieved with best parameters
        best_current: RMS current with best parameters
        history: Dictionary containing training history
    """
    # Initialize DAB converter
    dab = DABConverter()
    
    # Define state space: just target power for now (normalized to 0-1)
    state_space = [[0, dab.Pnom, 10]]  # Target power from 0 to Pnom, 10 discrete levels
    
    # Define action space: [D1, D2, D3]
    action_space = [
        [-dab.dt_max, dab.dt_max, 11],  # D1: phase shift between bridges, 11 levels
        [0, dab.dt_max, 6],             # D2: primary duty cycle, 6 levels
        [0, dab.dt_max, 6]              # D3: secondary duty cycle, 6 levels
    ]
    
    # Initialize Q-learning agent
    agent = QLearningAgent(state_space, action_space)
    
    # Prepare for tracking progress
    history = {
        'episode': [],
        'D1': [],
        'D2': [],
        'D3': [],
        'power': [],
        'target_power': [],
        'current': [],
        'reward': [],
        'exploration_rate': []
    }
    
    # Initialize best parameters
    best_params = None
    best_reward = -np.inf
    best_power = 0
    best_current = 0
    
    # Run Q-learning episodes
    print(f"Starting Q-learning optimization for target power: {target_power} W")
    print(f"Running {num_episodes} episodes...")
    
    for episode in tqdm(range(num_episodes)):
        # Current state is just the target power (normalized)
        current_state = [target_power]
        
        # Choose action (D1, D2, D3)
        action, action_indices = agent.choose_action(current_state)
        D1, D2, D3 = action
        
        # Apply action and observe reward
        _, actual_power = dab.calculate_power(D1, D2, D3)
        _, i_rms = dab.calculate_rms_current(D1, D2, D3)
        reward = dab.calculate_reward(D1, D2, D3, target_power)
        
        # Next state is the same (target power doesn't change)
        next_state = current_state
        
        # Update Q-value
        agent.update_q_value(current_state, action_indices, reward, next_state)
        
        # Track progress
        history['episode'].append(episode)
        history['D1'].append(D1)
        history['D2'].append(D2)
        history['D3'].append(D3)
        history['power'].append(actual_power)
        history['target_power'].append(target_power)
        history['current'].append(i_rms)
        history['reward'].append(reward)
        history['exploration_rate'].append(agent.exploration_rate)
        
        # Update best parameters if this is better
        if reward > best_reward:
            best_reward = reward
            best_params = (D1, D2, D3)
            best_power = actual_power
            best_current = i_rms
        
        # Decay exploration rate
        agent.decay_exploration()
    
    # Final evaluation with best parameters
    D1, D2, D3 = best_params
    _, best_power = dab.calculate_power(D1, D2, D3)
    _, best_current = dab.calculate_rms_current(D1, D2, D3)
    
    print("\nOptimization complete!")
    print(f"Best parameters found:")
    print(f"  D1 (phase shift between bridges): {D1:.4f}")
    print(f"  D2 (primary duty cycle): {D2:.4f}")
    print(f"  D3 (secondary duty cycle): {D3:.4f}")
    print(f"Resulting power: {best_power:.2f} W (target: {target_power} W)")
    print(f"RMS current: {best_current:.2f} A")
    
    # Visualize results if requested
    if visualize:
        visualize_results(history, best_params, dab)
    
    return best_params, best_power, best_current, history

def visualize_results(history, best_params, dab):
    """
    Visualize training results
    
    Args:
        history: Dictionary containing training history
        best_params: Best parameters found (D1, D2, D3)
        dab: DAB converter object
    """
    # Convert history to DataFrame for easier handling
    df = pd.DataFrame(history)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(18, 12))
    grid = plt.GridSpec(3, 3, hspace=0.3, wspace=0.3)
    
    # Plot 1: Phase shift parameters over episodes
    ax1 = fig.add_subplot(grid[0, :])
    ax1.plot(df['episode'], df['D1'], 'r-', label='D1 (Bridge phase shift)')
    ax1.plot(df['episode'], df['D2'], 'g-', label='D2 (Primary duty cycle)')
    ax1.plot(df['episode'], df['D3'], 'b-', label='D3 (Secondary duty cycle)')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Phase Shift (normalized)')
    ax1.set_title('Phase Shift Parameters During Training')
    ax1.legend()
    ax1.grid(True)
    
    # Plot 2: Power over episodes
    ax2 = fig.add_subplot(grid[1, :2])
    ax2.plot(df['episode'], df['power'], 'b-', label='Actual Power')
    ax2.plot(df['episode'], df['target_power'], 'r--', label='Target Power')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Power (W)')
    ax2.set_title('Power vs. Target During Training')
    ax2.legend()
    ax2.grid(True)
    
    # Plot 3: Current over episodes
    ax3 = fig.add_subplot(grid[1, 2])
    ax3.plot(df['episode'], df['current'], 'g-')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('RMS Current (A)')
    ax3.set_title('RMS Current During Training')
    ax3.grid(True)
    
    # Plot 4: Reward over episodes
    ax4 = fig.add_subplot(grid[2, 0])
    ax4.plot(df['episode'], df['reward'], 'purple')
    ax4.set_xlabel('Episode')
    ax4.set_ylabel('Reward')
    ax4.set_title('Reward During Training')
    ax4.grid(True)
    
    # Plot 5: Exploration rate over episodes
    ax5 = fig.add_subplot(grid[2, 1])
    ax5.plot(df['episode'], df['exploration_rate'], 'orange')
    ax5.set_xlabel('Episode')
    ax5.set_ylabel('Exploration Rate')
    ax5.set_title('Exploration Rate Decay')
    ax5.grid(True)
    
    # Plot 6: Power vs D1 heatmap (using latest D2, D3)
    ax6 = fig.add_subplot(grid[2, 2])
    
    # Sample D1 values for visualization
    D1_vals = np.linspace(-dab.dt_max, dab.dt_max, 100)
    
    # Use best D2 and D3 values
    _, _, D3 = best_params
    D2 = best_params[1]
    
    # Calculate power for each D1 value
    powers = []
    for D1 in D1_vals:
        _, p = dab.calculate_power(D1, D2, D3)
        powers.append(p)
    
    # Plot power vs D1
    ax6.plot(D1_vals, powers, 'b-')
    ax6.axvline(x=best_params[0], color='r', linestyle='--', label=f'Best D1: {best_params[0]:.3f}')
    ax6.set_xlabel('D1 (Bridge Phase Shift)')
    ax6.set_ylabel('Power (W)')
    ax6.set_title(f'Power vs D1 (with D2={D2:.3f}, D3={D3:.3f})')
    ax6.legend()
    ax6.grid(True)
    
    # Display the overall title
    plt.suptitle('DAB Converter TPS Optimization using Q-Learning', fontsize=16)
    
    # Adjust layout and save
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('dab_optimization_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create heatmap of D1-D2 power surface with fixed D3
    plt.figure(figsize=(12, 8))
    
    # Create meshgrid for D1 and D2
    D1_vals = np.linspace(-dab.dt_max, dab.dt_max, 50)
    D2_vals = np.linspace(0, dab.dt_max, 50)
    D1_mesh, D2_mesh = np.meshgrid(D1_vals, D2_vals)
    
    # Use best D3 value
    D3 = best_params[2]
    
    # Calculate power for each point in the grid
    power_grid = np.zeros_like(D1_mesh)
    for i in range(len(D2_vals)):
        for j in range(len(D1_vals)):
            _, power_grid[i, j] = dab.calculate_power(D1_mesh[i, j], D2_mesh[i, j], D3)
    
    # Create custom colormap
    colors = [(0, 0, 0.8), (0, 0.8, 0.8), (0, 0.8, 0), (0.8, 0.8, 0), (0.8, 0, 0)]
    cmap = LinearSegmentedColormap.from_list('custom_cmap', colors, N=100)
    
    # Plot heatmap
    plt.contourf(D1_mesh, D2_mesh, power_grid, 50, cmap=cmap)
    plt.colorbar(label='Power (W)')
    
    # Plot best point
    plt.plot(best_params[0], best_params[1], 'ro', markersize=10, label='Optimal Point')
    
    plt.xlabel('D1 (Bridge Phase Shift)')
    plt.ylabel('D2 (Primary Duty Cycle)')
    plt.title(f'Power Heatmap (D3 fixed at {D3:.3f})')
    plt.legend()
    plt.grid(True)
    plt.savefig('dab_power_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()

def sweep_target_powers():
    """Run optimization for multiple target powers and analyze results"""
    # Define target powers to test
    target_powers = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    
    # Prepare to store results
    results = {
        'target_power': [],
        'D1': [],
        'D2': [],
        'D3': [],
        'actual_power': [],
        'power_error': [],
        'rms_current': []
    }
    
    # Run optimization for each target power
    print("Running power sweep optimization...")
    for target in tqdm(target_powers):
        best_params, best_power, best_current, _ = run_optimization(target, num_episodes=500, visualize=False)
        
        # Store results
        results['target_power'].append(target)
        results['D1'].append(best_params[0])
        results['D2'].append(best_params[1])
        results['D3'].append(best_params[2])
        results['actual_power'].append(best_power)
        results['power_error'].append(abs(best_power - target))
        results['rms_current'].append(best_current)
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Display results
    print("\nOptimization results for various target powers:")
    print(results_df.to_string(index=False))
    
    # Create visualization of results
    plt.figure(figsize=(16, 12))
    
    # Plot 1: Phase Shifts vs Target Power
    plt.subplot(2, 2, 1)
    plt.plot(results['target_power'], results['D1'], 'r-o', label='D1')
    plt.plot(results['target_power'], results['D2'], 'g-o', label='D2')
    plt.plot(results['target_power'], results['D3'], 'b-o', label='D3')
    plt.xlabel('Target Power (W)')
    plt.ylabel('Phase Shift (normalized)')
    plt.title('Optimal Phase Shifts vs Target Power')
    plt.legend()
    plt.grid(True)
    
    # Plot 2: Actual Power vs Target Power
    plt.subplot(2, 2, 2)
    plt.plot(results['target_power'], results['actual_power'], 'b-o', label='Actual Power')
    plt.plot(results['target_power'], results['target_power'], 'r--', label='Target Power')
    plt.xlabel('Target Power (W)')
    plt.ylabel('Actual Power (W)')
    plt.title('Achieved Power vs Target Power')
    plt.legend()
    plt.grid(True)
    
    # Plot 3: Power Error vs Target Power
    plt.subplot(2, 2, 3)
    plt.bar(results['target_power'], results['power_error'], color='orange')
    plt.xlabel('Target Power (W)')
    plt.ylabel('Power Error (W)')
    plt.title('Power Error vs Target Power')
    plt.grid(True)
    
    # Plot 4: RMS Current vs Target Power
    plt.subplot(2, 2, 4)
    plt.plot(results['target_power'], results['rms_current'], 'g-o')
    plt.xlabel('Target Power (W)')
    plt.ylabel('RMS Current (A)')
    plt.title('RMS Current vs Target Power')
    plt.grid(True)
    
    plt.suptitle('DAB Converter Performance Across Power Range', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('dab_power_sweep.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Save results to CSV
    results_df.to_csv('dab_optimization_results.csv', index=False)
    print("Results saved to 'dab_optimization_results.csv'")
    
    return results_df

if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    
    print("DAB Converter TPS Optimization using Q-Learning")
    print("=" * 50)
    
    # Menu for user to choose what to run
    print("\nSelect an option:")
    print("1. Optimize TPS parameters for a single target power")
    print("2. Run power sweep optimization")
    
    choice = input("Enter your choice (1 or 2): ")
    
    if choice == '1':
        # Get target power from user
        target_power = float(input("\nEnter target power in Watts (100-1000): "))
        target_power = max(100, min(1000, target_power))  # Constrain to reasonable range
        
        # Get number of episodes
        num_episodes = int(input("Enter number of training episodes (500-5000): "))
        num_episodes = max(500, min(5000, num_episodes))  # Constrain to reasonable range
        
        # Run optimization
        best_params, best_power, best_current, _ = run_optimization(target_power, num_episodes, visualize=True)
        
    elif choice == '2':
        # Run power sweep
        results = sweep_target_powers()
        
    else:
        print("Invalid choice. Exiting.")