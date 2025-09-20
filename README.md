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

2. **Sample Code Snippet**  
dashboard.py (core snippet):
```bash
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

# Load sample data
data = pd.read_csv('data/sample_data.csv')

# Example visualization
plt.figure(figsize=(10,5))
plt.plot(data['Time'], data['Value'], label='Metric')
plt.xlabel('Time')
plt.ylabel('Value')
plt.title('Sample Metric Over Time')
plt.legend()
plt.show()
```
sample-dashboard.html (snippet):
```bash
<!-- DAB Converter TPS Optimization Dashboard Snippet -->
<!-- filepath: c:\Users\vaibh\OneDrive\Desktop\VS Code\IDP\sample_dashboard.html -->
<script src="https://cdnjs.cloudflare.com/ajax/libs/plotly.js/2.26.0/plotly.min.js"></script>
<script>
class DABConverter {
    constructor(V1, V2, Lk, fs, n, Pnom) {
        this.V1 = V1;
        this.V2 = V2;
        this.Lk = Lk * 1e-6;
        this.fs = fs * 1e3;
        this.n = n;
        this.Pnom = Pnom;
        this.dt_max = 0.5;
        this.Ts = 1 / this.fs;
        this.k = V2 / V1;
        this.Zbase = 8 * this.fs * this.Lk;
        this.Pbase = (this.V1 ** 2) / this.Zbase;
    }
    calculatePower(D1, D2, D3) {
        D1 = Math.max(-this.dt_max, Math.min(this.dt_max, D1));
        D2 = Math.max(0, Math.min(this.dt_max, D2));
        D3 = Math.max(0, Math.min(this.dt_max, D3));
        // ...power calculation logic...
        const normalized_power = this.k * D1; // simplified
        const p_actual = normalized_power * this.Pbase;
        return { normalized: normalized_power, actual: p_actual };
    }
}

// Example: Plotly chart for Power Tracking
Plotly.newPlot('powerChart', [{
    x: [0, 1, 2, 3, 4],
    y: [100, 200, 300, 400, 500],
    type: 'scatter',
    mode: 'lines+markers',
    name: 'Power (W)'
}], {
    title: 'Power Tracking',
    xaxis: { title: 'Episode' },
    yaxis: { title: 'Power (W)' }
});
</script>
<div id="powerChart" style="width:100%;height:400px;"></div>
```
## 🏁 Dashboard
<img width="1809" height="714" alt="Image" src="https://github.com/user-attachments/assets/763db87d-dee9-40d7-bea2-b400a38cc855" />
<img width="1769" height="559" alt="Image" src="https://github.com/user-attachments/assets/34219646-5b13-426a-9173-09a84f551784" />

## 🏁 Conclusion

- The Q-Learning-based TPS modulation successfully optimized the **DAB DC-DC converter** efficiency.  
- Achieved **99.6% power tracking accuracy** and **15.2% RMS current reduction**.  
- Demonstrated robust performance under varying load and operating conditions.

---

## 🔮 Future Work

- Integrate **real-time hardware-in-the-loop (HIL) testing** for physical converters.  
- Extend the model to **multi-objective optimization** including thermal and voltage stress.  
- Explore **deep reinforcement learning** methods for faster convergence and higher adaptability.  
- Develop a **web-based dashboard** to monitor converter performance dynamically.  






