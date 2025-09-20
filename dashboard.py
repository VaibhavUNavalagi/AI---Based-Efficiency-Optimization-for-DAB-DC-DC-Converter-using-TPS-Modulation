import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
import threading
from datetime import datetime
from matplotlib.colors import LinearSegmentedColormap
from tqdm import tqdm
import json

# Set page configuration
st.set_page_config(
    page_title="DAB DC-DC Converter Q-Learning Control",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #ff6b6b;
    }
    .success-card {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #28a745;
    }
    .info-card {
        background-color: #d1ecf1;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #17a2b8;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

class DABConverter:
    """Enhanced DAB Converter model with real-time capabilities"""
    
    def __init__(self, V1=100, V2=50, Lk=15e-6, fs=50e3, n=2, Pnom=1000, dt_max=0.5):
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
        
        # Real-time monitoring variables
        self.current_power = 0
        self.current_efficiency = 0
        self.temperature = 25  # Initial temperature
        
    def calculate_power(self, D1, D2, D3):
        """Calculate normalized power for given phase shift values"""
        D1 = np.clip(D1, -self.dt_max, self.dt_max)
        D2 = np.clip(D2, 0, self.dt_max)
        D3 = np.clip(D3, 0, self.dt_max)
        
        if D1 >= 0:
            if D1 <= D3 and D1 <= D2:
                p = D1 * (1 - D1) - (D2 - D1)**2/2 - (D3 - D1)**2/2
            elif D2 <= D1 and D2 <= D3:
                p = D2 * (1 - D2/2) + D1 * (D2 - D1) - (D3 - D1)**2/2
            elif D3 <= D1 and D3 <= D2:
                p = D3 * (1 - D3/2) + D1 * (D3 - D1) - (D2 - D1)**2/2
            else:
                p = D1 + D2 * (D3 - D2/2) + D3 * (D2 - D3/2)
        else:
            D1_abs = abs(D1)
            if D1_abs <= D3 and D1_abs <= D2:
                p = D1_abs * (1 - D1_abs) - (D2 - D1_abs)**2/2 - (D3 - D1_abs)**2/2
            elif D2 <= D1_abs and D2 <= D3:
                p = D2 * (1 - D2/2) + D1_abs * (D2 - D1_abs) - (D3 - D1_abs)**2/2
            elif D3 <= D1_abs and D3 <= D2:
                p = D3 * (1 - D3/2) + D1_abs * (D3 - D1_abs) - (D2 - D1_abs)**2/2
            else:
                p = D1_abs + D2 * (D3 - D2/2) + D3 * (D2 - D3/2)
            p = -p
            
        normalized_power = self.k * p
        p_actual = normalized_power * self.Pbase
        self.current_power = p_actual
        
        return normalized_power, p_actual
    
    def calculate_rms_current(self, D1, D2, D3):
        """Calculate RMS current with enhanced accuracy"""
        p_norm, _ = self.calculate_power(D1, D2, D3)
        
        # Enhanced RMS current calculation
        i_rms_squared = abs(p_norm) * (abs(D1) + D2 + D3) / 3
        i_rms_normalized = np.sqrt(i_rms_squared)
        
        i_base = self.V1 / self.Zbase
        i_rms_actual = i_rms_normalized * i_base
        
        return i_rms_normalized, i_rms_actual
    
    def calculate_efficiency(self, D1, D2, D3):
        """Calculate converter efficiency"""
        _, p_actual = self.calculate_power(D1, D2, D3)
        _, i_rms = self.calculate_rms_current(D1, D2, D3)
        
        # Simplified loss calculation
        conduction_loss = 0.1 * i_rms**2  # Conduction losses
        switching_loss = 0.05 * abs(p_actual) * (abs(D1) + D2 + D3) / 3  # Switching losses
        core_loss = 0.02 * abs(p_actual)  # Core losses
        
        total_loss = conduction_loss + switching_loss + core_loss
        
        if abs(p_actual) > 0:
            efficiency = (abs(p_actual) / (abs(p_actual) + total_loss)) * 100
        else:
            efficiency = 0
            
        self.current_efficiency = efficiency
        return efficiency, total_loss
    
    def calculate_reward(self, D1, D2, D3, target_power, weight_power=1.0, weight_current=0.5, weight_efficiency=0.8):
        """Enhanced reward function"""
        _, actual_power = self.calculate_power(D1, D2, D3)
        _, i_rms = self.calculate_rms_current(D1, D2, D3)
        efficiency, _ = self.calculate_efficiency(D1, D2, D3)
        
        power_error = abs(actual_power - target_power)
        power_error_norm = power_error / self.Pnom
        current_norm = i_rms / (self.V1 / self.Zbase)
        efficiency_norm = efficiency / 100
        
        power_reward = np.exp(-5 * power_error_norm)
        current_reward = np.exp(-2 * current_norm)
        efficiency_reward = efficiency_norm
        
        combined_reward = (weight_power * power_reward + 
                          weight_current * current_reward + 
                          weight_efficiency * efficiency_reward)
        
        return combined_reward

class EnhancedQLearningAgent:
    """Enhanced Q-Learning agent with real-time capabilities"""
    
    def __init__(self, state_space, action_space, learning_rate=0.1, discount_factor=0.9, 
                 exploration_rate=1.0, exploration_decay=0.995, min_exploration=0.01):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.min_exploration = min_exploration
        
        self.state_space = state_space
        self.action_space = action_space
        
        self.state_dims = [int(s[2]) for s in state_space]
        self.action_dims = [int(a[2]) for a in action_space]
        
        self.q_table = np.zeros(self.state_dims + self.action_dims)
        
        self.action_values = []
        for i, (min_val, max_val, num_actions) in enumerate(action_space):
            self.action_values.append(np.linspace(min_val, max_val, int(num_actions)))
        
        self.best_state = None
        self.best_action = None
        self.best_reward = -np.inf
        
        # Training history
        self.training_history = {
            'episodes': [],
            'rewards': [],
            'exploration_rates': [],
            'best_rewards': []
        }
    
    def state_to_index(self, state):
        """Convert continuous state to discrete index"""
        indices = []
        for i, (s, (min_val, max_val, num_states)) in enumerate(zip(state, self.state_space)):
            s_clipped = max(min_val, min(s, max_val))
            idx = int((s_clipped - min_val) / (max_val - min_val) * (num_states - 1))
            idx = max(0, min(idx, int(num_states) - 1))
            indices.append(idx)
        return tuple(indices)
    
    def choose_action(self, state):
        """Choose action using epsilon-greedy policy"""
        if np.random.rand() < self.exploration_rate:
            action_indices = [np.random.randint(0, dim) for dim in self.action_dims]
            action = [self.action_values[i][idx] for i, idx in enumerate(action_indices)]
            return action, tuple(action_indices)
        else:
            state_idx = self.state_to_index(state)
            action_indices = np.unravel_index(np.argmax(self.q_table[state_idx]), self.action_dims)
            action = [self.action_values[i][idx] for i, idx in enumerate(action_indices)]
            return action, action_indices
    
    def update_q_value(self, state, action_indices, reward, next_state):
        """Update Q-value using Q-learning algorithm"""
        state_idx = self.state_to_index(state)
        next_state_idx = self.state_to_index(next_state)
        
        current_q = self.q_table[state_idx + action_indices]
        max_next_q = np.max(self.q_table[next_state_idx])
        
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)
        self.q_table[state_idx + action_indices] = new_q
        
        if reward > self.best_reward:
            self.best_reward = reward
            self.best_state = state
            self.best_action = [self.action_values[i][idx] for i, idx in enumerate(action_indices)]
    
    def decay_exploration(self):
        """Decay exploration rate"""
        self.exploration_rate = max(self.min_exploration, self.exploration_rate * self.exploration_decay)

def main():
    """Main application function"""
    
    # Header
    st.title("⚡ Real-Time DAB DC-DC Converter Q-Learning Control System")
    st.markdown("---")
    
    # Initialize session state
    if 'dab_converter' not in st.session_state:
        st.session_state.dab_converter = None
    if 'agent' not in st.session_state:
        st.session_state.agent = None
    if 'training_complete' not in st.session_state:
        st.session_state.training_complete = False
    if 'real_time_data' not in st.session_state:
        st.session_state.real_time_data = {'time': [], 'power': [], 'efficiency': [], 'D1': [], 'D2': [], 'D3': []}
    
    # Sidebar configuration
    with st.sidebar:
        st.header("🔧 System Configuration")
        
        # Converter Parameters
        st.subheader("Converter Parameters")
        V1 = st.slider("Input Voltage V1 (V)", 50, 200, 100)
        V2 = st.slider("Output Voltage V2 (V)", 20, 100, 50)
        Lk = st.slider("Leakage Inductance (μH)", 5, 50, 15) * 1e-6
        fs = st.slider("Switching Frequency (kHz)", 20, 100, 50) * 1e3
        Pnom = st.slider("Nominal Power (W)", 500, 2000, 1000)
        
        # Q-learning Parameters
        st.subheader("Q-Learning Parameters")
        learning_rate = st.slider("Learning Rate", 0.01, 0.5, 0.1)
        discount_factor = st.slider("Discount Factor", 0.8, 0.99, 0.9)
        exploration_rate = st.slider("Initial Exploration Rate", 0.5, 1.0, 1.0)
        
        # Real-time Parameters
        st.subheader("Real-time Control")
        auto_update = st.checkbox("Auto Update", value=False)
        update_interval = st.slider("Update Interval (s)", 0.5, 5.0, 1.0)
        
        # Initialize system button
        if st.button("🚀 Initialize System", type="primary"):
            st.session_state.dab_converter = DABConverter(V1=V1, V2=V2, Lk=Lk, fs=fs, Pnom=Pnom)
            
            state_space = [[0, Pnom, 10]]
            action_space = [
                [-0.5, 0.5, 11],  # D1
                [0, 0.5, 6],      # D2
                [0, 0.5, 6]       # D3
            ]
            
            st.session_state.agent = EnhancedQLearningAgent(
                state_space, action_space, 
                learning_rate=learning_rate,
                discount_factor=discount_factor,
                exploration_rate=exploration_rate
            )
            
            st.success("✅ System initialized successfully!")
    
    # Main content area
    if st.session_state.dab_converter is None:
        st.info("👆 Please initialize the system using the sidebar controls")
        return
    
    # Create tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🎯 Training", "📊 Real-time Control", "📈 Performance Analysis", 
        "🔍 System Diagnostics", "⚙️ Advanced Settings"
    ])
    
    with tab1:
        st.header("Q-Learning Training")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Training parameters
            target_power = st.slider("Target Power (W)", 100, Pnom, int(Pnom*0.5))
            num_episodes = st.slider("Training Episodes", 100, 2000, 500)
            
            # Training progress area
            training_placeholder = st.empty()
            
            if st.button("🚀 Start Training", type="primary"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Training loop
                rewards_history = []
                power_history = []
                efficiency_history = []
                
                for episode in range(num_episodes):
                    # Simulate varying conditions
                    noise_factor = 0.1 * np.random.randn()
                    current_target = target_power * (1 + noise_factor)
                    
                    state = [current_target]
                    action, action_indices = st.session_state.agent.choose_action(state)
                    D1, D2, D3 = action
                    
                    reward = st.session_state.dab_converter.calculate_reward(D1, D2, D3, current_target)
                    _, actual_power = st.session_state.dab_converter.calculate_power(D1, D2, D3)
                    efficiency, _ = st.session_state.dab_converter.calculate_efficiency(D1, D2, D3)
                    
                    # Update Q-table
                    next_state = state  # Simplified for this example
                    st.session_state.agent.update_q_value(state, action_indices, reward, next_state)
                    st.session_state.agent.decay_exploration()
                    
                    # Store history
                    rewards_history.append(reward)
                    power_history.append(actual_power)
                    efficiency_history.append(efficiency)
                    
                    # Update progress
                    progress = (episode + 1) / num_episodes
                    progress_bar.progress(progress)
                    status_text.text(f"Episode {episode+1}/{num_episodes} - Reward: {reward:.3f}")
                
                st.session_state.training_complete = True
                st.session_state.training_history = {
                    'rewards': rewards_history,
                    'power': power_history,
                    'efficiency': efficiency_history
                }
                
                st.success("🎉 Training completed successfully!")
        
        with col2:
            st.subheader("Training Status")
            
            if st.session_state.training_complete:
                st.markdown('<div class="success-card">✅ Training Complete</div>', unsafe_allow_html=True)
                
                # Display training metrics
                final_reward = st.session_state.training_history['rewards'][-1]
                avg_efficiency = np.mean(st.session_state.training_history['efficiency'][-50:])
                
                st.metric("Final Reward", f"{final_reward:.3f}")
                st.metric("Avg Efficiency", f"{avg_efficiency:.1f}%")
                st.metric("Exploration Rate", f"{st.session_state.agent.exploration_rate:.3f}")
                
                # Training progress chart
                fig = go.Figure()
                episodes = list(range(len(st.session_state.training_history['rewards'])))
                
                fig.add_trace(go.Scatter(
                    x=episodes,
                    y=st.session_state.training_history['rewards'],
                    mode='lines',
                    name='Rewards',
                    line=dict(color='blue')
                ))
                
                fig.update_layout(
                    title="Training Progress",
                    xaxis_title="Episode",
                    yaxis_title="Reward",
                    height=300
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.markdown('<div class="info-card">⏳ Ready for Training</div>', unsafe_allow_html=True)
    
    with tab2:
        st.header("Real-time Control Dashboard")
        
        if not st.session_state.training_complete:
            st.warning("⚠️ Please complete training first")
            return
        
        # Real-time control interface
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            st.subheader("Control Parameters")
            
            # Real-time sliders
            rt_target_power = st.slider("Target Power (W)", 100, Pnom, int(Pnom*0.6), key="rt_power")
            rt_voltage_variation = st.slider("Voltage Variation (%)", 0, 20, 5)
            
            # Get optimal control action
            state = [rt_target_power]
            action, _ = st.session_state.agent.choose_action(state)
            D1_opt, D2_opt, D3_opt = action
            
            # Calculate performance metrics
            _, actual_power = st.session_state.dab_converter.calculate_power(D1_opt, D2_opt, D3_opt)
            efficiency, losses = st.session_state.dab_converter.calculate_efficiency(D1_opt, D2_opt, D3_opt)
            _, rms_current = st.session_state.dab_converter.calculate_rms_current(D1_opt, D2_opt, D3_opt)
            
            # Store real-time data
            current_time = time.time()
            st.session_state.real_time_data['time'].append(current_time)
            st.session_state.real_time_data['power'].append(actual_power)
            st.session_state.real_time_data['efficiency'].append(efficiency)
            st.session_state.real_time_data['D1'].append(D1_opt)
            st.session_state.real_time_data['D2'].append(D2_opt)
            st.session_state.real_time_data['D3'].append(D3_opt)
            
            # Keep only last 100 data points
            for key in st.session_state.real_time_data:
                if len(st.session_state.real_time_data[key]) > 100:
                    st.session_state.real_time_data[key] = st.session_state.real_time_data[key][-100:]
        
        with col2:
            st.subheader("Optimal Control")
            
            st.metric("D1 (Bridge Phase)", f"{D1_opt:.4f}")
            st.metric("D2 (Primary Duty)", f"{D2_opt:.4f}")
            st.metric("D3 (Secondary Duty)", f"{D3_opt:.4f}")
        
        with col3:
            st.subheader("Performance")
            
            power_error = abs(actual_power - rt_target_power)
            st.metric("Actual Power", f"{actual_power:.1f} W", f"{power_error:.1f} W")
            st.metric("Efficiency", f"{efficiency:.1f}%")
            st.metric("RMS Current", f"{rms_current:.2f} A")
        
        # Real-time plots
        if len(st.session_state.real_time_data['time']) > 1:
            # Create subplot figure
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Power Tracking', 'Efficiency', 'Phase Shifts', 'Current'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            time_data = st.session_state.real_time_data['time']
            time_relative = [(t - time_data[0]) for t in time_data]
            
            # Power tracking
            fig.add_trace(
                go.Scatter(x=time_relative, y=st.session_state.real_time_data['power'], 
                          name="Actual Power", line=dict(color="blue")),
                row=1, col=1
            )
            
            # Efficiency
            fig.add_trace(
                go.Scatter(x=time_relative, y=st.session_state.real_time_data['efficiency'], 
                          name="Efficiency", line=dict(color="green")),
                row=1, col=2
            )
            
            # Phase shifts
            fig.add_trace(
                go.Scatter(x=time_relative, y=st.session_state.real_time_data['D1'], 
                          name="D1", line=dict(color="red")),
                row=2, col=1
            )
            fig.add_trace(
                go.Scatter(x=time_relative, y=st.session_state.real_time_data['D2'], 
                          name="D2", line=dict(color="green")),
                row=2, col=1
            )
            fig.add_trace(
                go.Scatter(x=time_relative, y=st.session_state.real_time_data['D3'], 
                          name="D3", line=dict(color="blue")),
                row=2, col=1
            )
            
            fig.update_layout(height=600, showlegend=False, title_text="Real-time System Response")
            st.plotly_chart(fig, use_container_width=True)
        
        # Auto-update functionality
        if auto_update:
            time.sleep(update_interval)
            st.rerun()
    
    with tab3:
        st.header("Performance Analysis")
        
        if not st.session_state.training_complete:
            st.warning("⚠️ Please complete training first")
            return
        
        # Performance analysis tools
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Efficiency Mapping")
            
            # Create efficiency heatmap
            power_range = np.linspace(100, Pnom, 20)
            voltage_range = np.linspace(0.8, 1.2, 20)
            
            efficiency_map = np.zeros((len(power_range), len(voltage_range)))
            
            for i, power in enumerate(power_range):
                for j, v_factor in enumerate(voltage_range):
                    state = [power]
                    action, _ = st.session_state.agent.choose_action(state)
                    D1, D2, D3 = action
                    
                    # Temporarily modify voltage
                    orig_V1 = st.session_state.dab_converter.V1
                    st.session_state.dab_converter.V1 = orig_V1 * v_factor
                    
                    eff, _ = st.session_state.dab_converter.calculate_efficiency(D1, D2, D3)
                    efficiency_map[i, j] = eff
                    
                    # Restore original voltage
                    st.session_state.dab_converter.V1 = orig_V1
            
            fig = go.Figure(data=go.Heatmap(
                z=efficiency_map,
                x=voltage_range,
                y=power_range,
                colorscale='RdYlGn',
                colorbar=dict(title="Efficiency (%)")
            ))
            
            fig.update_layout(
                title="Efficiency Map",
                xaxis_title="Voltage Factor",
                yaxis_title="Power (W)"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Power Sweep Analysis")
            
            # Perform power sweep
            test_powers = np.linspace(100, Pnom, 20)
            sweep_results = {'power': [], 'efficiency': [], 'D1': [], 'D2': [], 'D3': []}
            
            for power in test_powers:
                state = [power]
                action, _ = st.session_state.agent.choose_action(state)
                D1, D2, D3 = action
                
                eff, _ = st.session_state.dab_converter.calculate_efficiency(D1, D2, D3)
                
                sweep_results['power'].append(power)
                sweep_results['efficiency'].append(eff)
                sweep_results['D1'].append(D1)
                sweep_results['D2'].append(D2)
                sweep_results['D3'].append(D3)
            
            # Plot results
            fig = make_subplots(rows=2, cols=1, subplot_titles=('Efficiency vs Power', 'Phase Shifts vs Power'))
            
            fig.add_trace(
                go.Scatter(x=sweep_results['power'], y=sweep_results['efficiency'], 
                          name="Efficiency", line=dict(color="green")),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(x=sweep_results['power'], y=sweep_results['D1'], 
                          name="D1", line=dict(color="red")),
                row=2, col=1
            )
            fig.add_trace(
                go.Scatter(x=sweep_results['power'], y=sweep_results['D2'], 
                          name="D2", line=dict(color="green")),
                row=2, col=1
            )
            fig.add_trace(
                go.Scatter(x=sweep_results['power'], y=sweep_results['D3'], 
                          name="D3", line=dict(color="blue")),
                row=2, col=1
            )
            
            fig.update_layout(height=600, title_text="Power Sweep Analysis")
            st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.header("System Diagnostics")
        
        # System health monitoring
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("System Status")
            
            # Health indicators
            if st.session_state.training_complete:
                st.markdown('<div class="success-card">🟢 Training: Complete</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="info-card">🟡 Training: Pending</div>', unsafe_allow_html=True)
            
            if st.session_state.dab_converter:
                st.markdown('<div class="success-card">🟢 Converter: Online</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="metric-card">🔴 Converter: Offline</div>', unsafe_allow_html=True)
            
            # Q-table statistics
            if st.session_state.agent:
                q_table = st.session_state.agent.q_table
                st.metric("Q-table Size", f"{q_table.size:,}")
                st.metric("Non-zero Entries", f"{np.count_nonzero(q_table):,}")
                st.metric("Max Q-value", f"{np.max(q_table):.3f}")
        
        with col2:
            st.subheader("Performance Metrics")
            
            if st.session_state.training_complete and len(st.session_state.real_time_data['efficiency']) > 0:
                avg_eff = np.mean(st.session_state.real_time_data['efficiency'][-10:])
                std_eff = np.std(st.session_state.real_time_data['efficiency'][-10:])
                
                st.metric("Average Efficiency", f"{avg_eff:.1f}%")
                st.metric("Efficiency Std Dev", f"{std_eff:.2f}%")
                
                if len(st.session_state.real_time_data['power']) > 1:
                    power_stability = np.std(st.session_state.real_time_data['power'][-10:])
                    st.metric("Power Stability", f"{power_stability:.1f}W")
        
        with col3:
            st.subheader("System Parameters")
            
            if st.session_state.dab_converter:
                st.text(f"V1: {st.session_state.dab_converter.V1}V")
                st.text(f"V2: {st.session_state.dab_converter.V2}V")
                st.text(f"Frequency: {st.session_state.dab_converter.fs/1000:.0f}kHz")
                st.text(f"Nominal Power: {st.session_state.dab_converter.Pnom}W")
        
        # Diagnostic plots
        if st.session_state.training_complete:
            st.subheader("Training Convergence Analysis")
            
            # Moving average of rewards
            rewards = st.session_state.training_history['rewards']
            window_size = min(50, len(rewards)//4)
            
            if len(rewards) > window_size:
                moving_avg = pd.Series(rewards).rolling(window=window_size).mean()
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=list(range(len(rewards))),
                    y=rewards,
                    mode='lines',
                    name='Raw Rewards',
                    line=dict(color='lightblue', width=1)
                ))
                fig.add_trace(go.Scatter(
                    x=list(range(len(moving_avg))),
                    y=moving_avg,
                    mode='lines',
                    name=f'Moving Average ({window_size})',
                    line=dict(color='blue', width=2)
                ))
                
                fig.update_layout(
                    title="Training Convergence",
                    xaxis_title="Episode",
                    yaxis_title="Reward"
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    with tab5:
        st.header("Advanced Settings & Export")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Export Options")
            
            if st.session_state.training_complete:
                # Export training data
                if st.button("📊 Export Training Data"):
                    training_df = pd.DataFrame({
                        'Episode': range(len(st.session_state.training_history['rewards'])),
                        'Reward': st.session_state.training_history['rewards'],
                        'Power': st.session_state.training_history['power'],
                        'Efficiency': st.session_state.training_history['efficiency']
                    })
                    
                    csv = training_df.to_csv(index=False)
                    st.download_button(
                        label="Download Training Data CSV",
                        data=csv,
                        file_name=f"dab_training_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                
                # Export Q-table
                if st.button("🧠 Export Q-table"):
                    q_table_flat = st.session_state.agent.q_table.flatten()
                    q_df = pd.DataFrame({'Q_values': q_table_flat})
                    
                    csv = q_df.to_csv(index=False)
                    st.download_button(
                        label="Download Q-table CSV",
                        data=csv,
                        file_name=f"dab_qtable_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
        
        with col2:
            st.subheader("System Configuration")
            
            # Reset options
            if st.button("🔄 Reset Training", type="secondary"):
                st.session_state.training_complete = False
                st.session_state.training_history = {}
                if st.session_state.agent:
                    st.session_state.agent.q_table.fill(0)
                    st.session_state.agent.exploration_rate = 1.0
                st.success("Training reset complete!")
            
            if st.button("🧹 Clear Real-time Data"):
                st.session_state.real_time_data = {
                    'time': [], 'power': [], 'efficiency': [], 
                    'D1': [], 'D2': [], 'D3': []
                }
                st.success("Real-time data cleared!")
            
            # Advanced parameters
            st.subheader("Advanced Parameters")
            
            if st.session_state.agent:
                new_lr = st.slider("Adjust Learning Rate", 0.01, 0.5, st.session_state.agent.learning_rate)
                new_df = st.slider("Adjust Discount Factor", 0.8, 0.99, st.session_state.agent.discount_factor)
                
                if st.button("Update Parameters"):
                    st.session_state.agent.learning_rate = new_lr
                    st.session_state.agent.discount_factor = new_df
                    st.success("Parameters updated!")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    ### 🚀 Real-Time DAB DC-DC Converter Q-Learning Control System
    
    This advanced Streamlit application demonstrates **real-time Q-learning control** for Dual Active Bridge (DAB) DC-DC converters with Triple Phase-Shift (TPS) modulation[6][7]. 
    
    **Key Features:**
    - **🎯 Interactive Training**: Train Q-learning agents with customizable parameters
    - **📊 Real-time Control**: Live control dashboard with auto-updating metrics
    - **📈 Performance Analysis**: Comprehensive efficiency mapping and power sweep analysis
    - **🔍 System Diagnostics**: Health monitoring and convergence analysis
    - **⚙️ Advanced Settings**: Data export and parameter tuning capabilities
    
    **Technical Implementation:**
    - Enhanced DAB converter model with accurate power and efficiency calculations
    - Real-time Q-learning agent with epsilon-greedy exploration
    - Interactive visualizations using Plotly for superior user experience
    - Comprehensive performance monitoring and diagnostic tools
    
    The system optimizes phase-shift angles (D1, D2, D3) to achieve maximum power efficiency while maintaining precise power control across varying operating conditions[6].
    """)

if __name__ == "__main__":
    main()
