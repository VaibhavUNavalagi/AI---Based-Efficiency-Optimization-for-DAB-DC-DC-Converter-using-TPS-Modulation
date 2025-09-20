# AI-Based Efficiency Optimization for DAB DC-DC Converter using TPS Modulation

[![GitHub](https://img.shields.io/badge/GitHub-Repository-blue?logo=github)](https://github.com/VaibhavUNavalagi/AI-Based-Efficiency-Optimization-for-DAB-DC-DC-Converter-using-TPS-Modulation)

## Overview
This project leverages **Q-Learning**, a reinforcement learning technique, to optimize the efficiency of a **Dual Active Bridge (DAB) DC-DC Converter** using **Triple Phase Shift (TPS) modulation**. The approach focuses on improving power control while reducing losses and enhancing system stability.

- Achieved **99.6%** power tracking accuracy.
- Reduced RMS current by **15.2%**.
- Ensured robust converter performance under varying load conditions.

## Features
- Reinforcement learning-based control for DAB converter.
- Adaptive TPS modulation for efficiency optimization.
- Real-time simulation and performance monitoring.

## 🔹 Installation and build

1. **Install dependencies:**  

```bash
# Install numpy
pip install numpy

# Install matplotlib
pip install matplotlib

# Install pandas
pip install pandas

# Install tqdm
pip install tqdm
```

2. **Running the Project:**  
Run the main Python file:
```bash
python q-learning-algo.py
```

3. **Sample Code Snippet:**  
Below is a simplified version of the core Q-Learning loop used in this project:
```bash
import numpy as np

# Initialize Q-Table
state_size = 100
action_size = 5
Q = np.zeros((state_size, action_size))

# Q-Learning parameters
learning_rate = 0.01
discount_factor = 0.9
epsilon = 0.1
episodes = 1000

for episode in range(episodes):
    state = env.reset()
    done = False

    while not done:
        # ε-greedy action selection
        action = np.random.randint(action_size) if np.random.rand() < epsilon else np.argmax(Q[state])
        
        # Take action and observe reward
        next_state, reward, done = env.step(action)
        
        # Update Q-Table
        Q[state, action] += learning_rate * (reward + discount_factor * np.max(Q[next_state]) - Q[state, action])
        state = next_state

# Save Q-Table
np.save('q_table.npy', Q)
```






