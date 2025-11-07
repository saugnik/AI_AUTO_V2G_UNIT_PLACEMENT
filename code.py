import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import warnings
import pickle
from collections import deque
import random
import os
from datetime import datetime
warnings.filterwarnings('ignore')

class V2GEnvironment:
    """Enhanced Environment for V2G decision making with reduced HOLD bias"""
    def __init__(self, car_session, tariff_df, current_slot):
        self.car_session = car_session
        self.tariff_df = tariff_df
        self.current_slot = current_slot
        self.min_power_kw = 3.0
        self.max_power_kw = 22.0
        self.min_energy_per_slot = 1.5  # kWh
        self.max_energy_per_slot = 11.0  # kWh
        
    def get_state(self):
        """Get current state for the RL agent"""
        car = self.car_session
        current_tariff = self.tariff_df[self.tariff_df['slot'] == self.current_slot].iloc[0]
        
        # Calculate features
        departure_slot = car['departure_slot']
        slots_remaining = departure_slot - self.current_slot if departure_slot <= 48 else (departure_slot - self.current_slot) % 48
        
        energy_needed = max(0, (car['required_soc'] - car['current_soc']) * car['battery_capacity'])
        available_for_v2g = max(0, (car['current_soc'] - max(car['required_soc'] + 0.05, 0.20)) * car['battery_capacity'])
        
        # Enhanced state features with more decision-driving information
        state = np.array([
            car['current_soc'],  # Current SoC
            car['required_soc'],  # Required SoC
            energy_needed / car['battery_capacity'],  # Energy needed (normalized)
            available_for_v2g / car['battery_capacity'],  # Available for V2G (normalized)
            current_tariff['rate_inr_per_kwh'] / 14.0,  # Normalized rate (max 14 INR/kWh)
            current_tariff['grid_demand_factor'],  # Grid demand
            slots_remaining / 48.0,  # Time remaining (normalized)
            (self.current_slot % 48) / 48.0,  # Time of day (normalized)
            car['battery_capacity'] / 100.0,  # Battery size (normalized)
            min(1.0, energy_needed / (slots_remaining * 5.0 + 0.1)),  # Charging urgency
            min(1.0, available_for_v2g / 10.0),  # V2G opportunity
        ])
        
        return state
    
    def step(self, action):
        """Execute action and return reward with enhanced reward structure"""
        car = self.car_session
        current_tariff = self.tariff_df[self.tariff_df['slot'] == self.current_slot].iloc[0]
        rate = current_tariff['rate_inr_per_kwh']
        grid_demand = current_tariff['grid_demand_factor']
        
        action_name = 'HOLD'
        power_kw = 0
        energy_kwh = 0
        reward = 0
        
        departure_slot = car['departure_slot']
        slots_remaining = departure_slot - self.current_slot if departure_slot <= 48 else (departure_slot - self.current_slot) % 48
        
        energy_needed = max(0, (car['required_soc'] - car['current_soc']) * car['battery_capacity'])
        available_for_v2g = max(0, (car['current_soc'] - max(car['required_soc'] + 0.05, 0.20)) * car['battery_capacity'])
        
        if action == 1:  # CHARGE_LOW
            if energy_needed > 0.5 or (car['current_soc'] < 0.85 and rate < 10):
                energy_kwh = min(4.0, max(1.5, energy_needed * 0.3), (1.0 - car['current_soc']) * car['battery_capacity'])
                if energy_kwh >= self.min_energy_per_slot:
                    action_name = 'CHARGE'
                    power_kw = min(22.0, max(3.0, energy_kwh * 2))
                    cost = energy_kwh * rate
                    reward = -cost + (20 - rate) * 2  
                    
        elif action == 2:  # CHARGE_MED
            if energy_needed > 1.0 or (car['current_soc'] < 0.90 and rate < 8):
                energy_kwh = min(7.0, max(2.0, energy_needed * 0.6), (1.0 - car['current_soc']) * car['battery_capacity'])
                if energy_kwh >= self.min_energy_per_slot:
                    action_name = 'CHARGE'
                    power_kw = min(22.0, max(3.0, energy_kwh * 2))
                    cost = energy_kwh * rate
                    reward = -cost + (18 - rate) * 1.5
                    
        elif action == 3:  # CHARGE_HIGH
            if energy_needed > 2.0 or (car['current_soc'] < 0.75 and slots_remaining < 5):
                energy_kwh = min(11.0, max(3.0, energy_needed * 0.8), (1.0 - car['current_soc']) * car['battery_capacity'])
                if energy_kwh >= self.min_energy_per_slot:
                    action_name = 'CHARGE'
                    power_kw = min(22.0, max(3.0, energy_kwh * 2))
                    cost = energy_kwh * rate
                    reward = -cost + (15 - rate) * 1.0
                    
        elif action == 4:  # DISCHARGE_LOW
            if available_for_v2g >= 2.0 and slots_remaining > 2 and rate > 8:
                energy_kwh = -min(4.0, available_for_v2g * 0.3)
                if abs(energy_kwh) >= self.min_energy_per_slot:
                    action_name = 'DISCHARGE'
                    power_kw = max(-22.0, min(-3.0, energy_kwh * 2))
                    revenue = abs(energy_kwh) * rate * 0.90
                    reward = revenue + (rate - 8) * 5 + (grid_demand * 30)
                    
        elif action == 5:  # DISCHARGE_MED
            if available_for_v2g >= 3.0 and slots_remaining > 2 and rate > 10:
                energy_kwh = -min(7.0, available_for_v2g * 0.5)
                if abs(energy_kwh) >= self.min_energy_per_slot:
                    action_name = 'DISCHARGE'
                    power_kw = max(-22.0, min(-3.0, energy_kwh * 2))
                    revenue = abs(energy_kwh) * rate * 0.90
                    reward = revenue + (rate - 8) * 4 + (grid_demand * 40)
                    
        elif action == 6:  # DISCHARGE_HIGH
            if available_for_v2g >= 5.0 and slots_remaining > 3 and rate > 12:
                energy_kwh = -min(11.0, available_for_v2g * 0.7)
                if abs(energy_kwh) >= self.min_energy_per_slot:
                    action_name = 'DISCHARGE'
                    power_kw = max(-22.0, min(-3.0, energy_kwh * 2))
                    revenue = abs(energy_kwh) * rate * 0.90
                    reward = revenue + (rate - 8) * 3 + (grid_demand * 50)
        

        if action_name == 'HOLD':
            # Heavy penalty for not acting when there's clear opportunity
            if energy_needed > 2.0 and rate < 6 and slots_remaining > 1:
                reward = -15  # Should charge at low rates
            elif available_for_v2g > 3.0 and rate > 11 and slots_remaining > 2:
                reward = -20  # Should discharge at high rates
            elif energy_needed > 1.0 and slots_remaining < 3:
                reward = -25  # Urgently need to charge
            elif rate < 4 and car['current_soc'] < 0.9:
                reward = -10  # Missing cheap charging opportunity
            elif rate > 13 and available_for_v2g > 2:
                reward = -18  # Missing expensive discharge opportunity
            else:
                reward = -2   # Small penalty for any hold to encourage action
        
        # Power constraint validation
        if action_name != 'HOLD':
            abs_power = abs(power_kw)
            if abs_power < self.min_power_kw or abs_power > self.max_power_kw:
                reward = -100  # Heavy penalty for constraint violation
                action_name = 'HOLD'
                power_kw = 0
                energy_kwh = 0
        
        return {
            'action': action_name,
            'power_kw': round(power_kw, 3),
            'energy_kwh': round(energy_kwh, 4),
            'reward': reward,
            'rate_inr_per_kwh': rate,
            'grid_demand_factor': grid_demand
        }

class DQNAgent:
    """Enhanced Deep Q-Network agent with better exploration and model persistence"""
    def __init__(self, state_size=11, action_size=7, learning_rate=0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=15000)  # Increased memory
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.05  # Higher minimum exploration
        self.epsilon_decay = 0.992  # Slower decay
        self.learning_rate = learning_rate
        
        # Enhanced Q-table with action bias counters
        self.q_table = {}
        self.action_counts = np.zeros(action_size)
        self.training_history = []
        
    def discretize_state(self, state):
        """Convert continuous state to discrete for Q-table"""
        # More granular discretization to reduce HOLD bias
        discrete_state = tuple([round(s, 1) for s in state])
        return discrete_state
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        """Enhanced action selection with bias correction"""
        discrete_state = self.discretize_state(state)
        
        # Higher exploration rate for better action diversity
        exploration_rate = max(self.epsilon, 0.15)
        
        if np.random.random() <= exploration_rate:
            # Biased random selection to reduce HOLD preference
            action_probs = np.array([0.1, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15])  # Less HOLD probability
            return np.random.choice(self.action_size, p=action_probs)
        
        if discrete_state not in self.q_table:
            self.q_table[discrete_state] = np.random.uniform(-1, 1, self.action_size)  # Random initialization
            
        q_values = self.q_table[discrete_state].copy()
        
        # Apply action balancing - penalize overused actions
        total_actions = max(np.sum(self.action_counts), 1)
        for i in range(self.action_size):
            action_ratio = self.action_counts[i] / total_actions
            if action_ratio > 0.3:  # If action used more than 30%
                q_values[i] -= action_ratio * 10  # Penalty
        
        selected_action = np.argmax(q_values)
        self.action_counts[selected_action] += 1
        
        return selected_action
    
    def replay(self, batch_size=64):  # Increased batch size
        """Enhanced training with better updates"""
        if len(self.memory) < batch_size:
            return
            
        batch = random.sample(self.memory, batch_size)
        
        for state, action, reward, next_state, done in batch:
            discrete_state = self.discretize_state(state)
            discrete_next_state = self.discretize_state(next_state)
            
            if discrete_state not in self.q_table:
                self.q_table[discrete_state] = np.random.uniform(-1, 1, self.action_size)
            if discrete_next_state not in self.q_table:
                self.q_table[discrete_next_state] = np.random.uniform(-1, 1, self.action_size)
            
            target = reward
            if not done:
                target = reward + 0.95 * np.amax(self.q_table[discrete_next_state])
            
            # Enhanced Q-learning update with learning rate
            current_q = self.q_table[discrete_state][action]
            self.q_table[discrete_state][action] = current_q + self.learning_rate * (target - current_q)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def save_model(self, filepath):
        """Save the trained model and training data"""
        model_data = {
            'q_table': self.q_table,
            'action_counts': self.action_counts,
            'epsilon': self.epsilon,
            'training_history': self.training_history,
            'state_size': self.state_size,
            'action_size': self.action_size,
            'learning_rate': self.learning_rate,
            'memory': list(self.memory),
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"✅ Model saved to: {filepath}")
        print(f"   Q-table size: {len(self.q_table)} states")
        print(f"   Memory size: {len(self.memory)} experiences")
        print(f"   Final epsilon: {self.epsilon:.4f}")
    
    def load_model(self, filepath):
        """Load a pre-trained model"""
        if not os.path.exists(filepath):
            print(f"❌ Model file not found: {filepath}")
            return False
        
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.q_table = model_data['q_table']
            self.action_counts = model_data['action_counts']
            self.epsilon = model_data['epsilon']
            self.training_history = model_data.get('training_history', [])
            self.memory = deque(model_data.get('memory', []), maxlen=15000)
            
            print(f"✅ Model loaded from: {filepath}")
            print(f"   Trained on: {model_data.get('timestamp', 'Unknown')}")
            print(f"   Q-table size: {len(self.q_table)} states")
            print(f"   Memory size: {len(self.memory)} experiences")
            print(f"   Loaded epsilon: {self.epsilon:.4f}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {str(e)}")
            return False

class RLBasedV2GGridManager:
    def __init__(self):
        self.cars_data = []
        self.hourly_loads = []
        self.individual_car_logs = []
        self.all_cars_all_slots = []
        self.financial_summary = []
        self.agent = DQNAgent()
        self.training_dataset = None
        self.model_dir = "trained_models"
        
        # Create models directory if it doesn't exist
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        
    def time_to_slot(self, hour, minute):
        """Convert hour:minute to 30-minute slot index (0-47)"""
        return int((hour * 2) + (minute // 30))
    
    def slot_to_time(self, slot):
        """Convert slot index to hour:minute format"""
        hour = slot // 2
        minute = (slot % 2) * 30
        return f"{hour:02d}:{minute:02d}"
        
    def generate_v2g_optimized_data(self, n_cars=100):
        """Generate EV data specifically optimized for significant V2G participation"""
        print(f"Generating V2G-maximized data for {n_cars} EVs...")
        
        np.random.seed(42)
        ev_sessions = []
        
        variance_groups = n_cars // 5 
        
        for group in range(variance_groups):
            base_arrival_hour = 16 + (group % 6)  
            base_departure_hour = 7 + (group % 4)  
            
            for car_in_group in range(5):
                car_id = group * 5 + car_in_group + 1
                
                arrival_hour = base_arrival_hour + np.random.randint(-1, 2)
                arrival_hour = max(15, min(22, arrival_hour))  
                arrival_min = np.random.choice([0, 30])
                arrival_slot = self.time_to_slot(arrival_hour, arrival_min)
                
                departure_hour = base_departure_hour + np.random.randint(-1, 2)
                departure_hour = max(6, min(11, departure_hour)) 
                departure_min = np.random.choice([0, 30])
                departure_slot = self.time_to_slot(departure_hour, departure_min)
                
                if departure_slot <= arrival_slot:
                    departure_slot += 48
                
                battery_capacity = np.random.choice([40, 50, 60, 75, 85], 
                                                  p=[0.05, 0.20, 0.35, 0.25, 0.15])
                
                initial_soc = np.random.uniform(0.65, 0.95)  # Higher initial SoC for more V2G
                required_soc = np.random.uniform(0.60, 0.75)  # Lower required SoC
                
                if car_in_group == 0:  
                    battery_capacity += 10
                    initial_soc = min(0.95, initial_soc + 0.10)
                elif car_in_group == 4: 
                    battery_capacity = max(30, battery_capacity - 5)
                    initial_soc = max(0.60, initial_soc - 0.05)
                
                ev_sessions.append({
                    'car_id': car_id,
                    'group_id': group + 1,
                    'arrival_slot': arrival_slot,
                    'departure_slot': departure_slot,
                    'initial_soc': initial_soc,
                    'required_soc': required_soc,
                    'battery_capacity': battery_capacity,
                    'arrival_time': f"{arrival_hour:02d}:{arrival_min:02d}",
                    'departure_time': f"{departure_hour:02d}:{departure_min:02d}",
                    'current_soc': initial_soc,
                    'total_earnings': 0.0,  
                    'total_costs': 0.0      
                })
        
        return ev_sessions
    
    def generate_dynamic_tariff(self):
        """Generate tariff with 3-14 INR/kWh range"""
        tariff_data = []
        
        for slot in range(48):
            hour = slot // 2
            
            # Adjust rate ranges to 3-14 INR/kWh
            if 6 <= hour <= 9:  # Morning peak
                base_rate = np.random.uniform(10.0, 14.0)
                grid_demand = np.random.uniform(0.80, 0.95)
            elif 18 <= hour <= 22:  # Evening peak
                base_rate = np.random.uniform(11.0, 14.0)  
                grid_demand = np.random.uniform(0.85, 1.0)
            elif 0 <= hour <= 5:  # Night valley
                base_rate = np.random.uniform(3.0, 5.0)   
                grid_demand = np.random.uniform(0.15, 0.35)
            elif 13 <= hour <= 16:  # Afternoon valley
                base_rate = np.random.uniform(4.0, 6.0)  
                grid_demand = np.random.uniform(0.40, 0.60)
            else:  # Other hours
                base_rate = np.random.uniform(6.0, 9.0)
                grid_demand = np.random.uniform(0.55, 0.75)
            
            rate = np.clip(base_rate + np.random.normal(0, 0.5), 3.0, 14.0)
            demand = np.clip(grid_demand + np.random.normal(0, 0.03), 0.1, 1.0)
            
            tariff_data.append({
                'slot': slot,
                'time': self.slot_to_time(slot),
                'rate_inr_per_kwh': round(rate, 2),
                'grid_demand_factor': round(demand, 3),
                'base_grid_load_kw': 0  
            })
        
        return pd.DataFrame(tariff_data)
    
    def save_training_dataset(self, ev_sessions, tariff_df, filename_prefix="training_data"):
        """Save the training dataset"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save EV sessions
        ev_df = pd.DataFrame(ev_sessions)
        ev_file = f"{self.model_dir}/{filename_prefix}_ev_sessions_{timestamp}.xlsx"
        ev_df.to_excel(ev_file, index=False)
        
        # Save tariff data
        tariff_file = f"{self.model_dir}/{filename_prefix}_tariff_{timestamp}.xlsx"
        tariff_df.to_excel(tariff_file, index=False)
        
        # Save combined dataset info
        dataset_info = {
            'ev_sessions_file': ev_file,
            'tariff_file': tariff_file,
            'n_cars': len(ev_sessions),
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'tariff_range': f"{tariff_df['rate_inr_per_kwh'].min():.2f} - {tariff_df['rate_inr_per_kwh'].max():.2f} INR/kWh"
        }
        
        dataset_info_file = f"{self.model_dir}/{filename_prefix}_info_{timestamp}.pkl"
        with open(dataset_info_file, 'wb') as f:
            pickle.dump(dataset_info, f)
        
        print(f"\n📁 TRAINING DATASET SAVED:")
        print(f"   🚗 EV Sessions: {ev_file}")
        print(f"   💰 Tariff Data: {tariff_file}")
        print(f"   ℹ️  Dataset Info: {dataset_info_file}")
        
        self.training_dataset = dataset_info
        return dataset_info
    
    def train_agent_on_historical_data(self, ev_sessions, tariff_df, episodes=200):
        """Enhanced training with more episodes and model saving"""
        print(f"Training RL agent for {episodes} episodes...")
        
        episode_rewards = []
        
        for episode in range(episodes):
            total_reward = 0
            
            # Train on more scenarios per episode
            for _ in range(60):  
                car = random.choice(ev_sessions)
                slot = random.randint(car['arrival_slot'], min(car['departure_slot']-1, 47))
                
                # Create more varied temporary car states
                temp_car = car.copy()
                soc_variation = 0.15 if episode > 100 else 0.1
                temp_car['current_soc'] = np.clip(
                    np.random.uniform(car['initial_soc'] - soc_variation, 
                                    car['initial_soc'] + soc_variation), 0.2, 1.0)
                
                env = V2GEnvironment(temp_car, tariff_df, slot)
                state = env.get_state()
                
                action = self.agent.act(state)
                result = env.step(action)
                reward = result['reward']
                
                # Simulate next state
                next_slot = min(slot + 1, 47)
                next_env = V2GEnvironment(temp_car, tariff_df, next_slot)
                next_state = next_env.get_state()
                
                done = (next_slot >= temp_car['departure_slot'] - 1)
                
                self.agent.remember(state, action, reward, next_state, done)
                total_reward += reward
            
            # Train the agent
            self.agent.replay()
            episode_rewards.append(total_reward)
            
            # Store training history
            self.agent.training_history.append({
                'episode': episode,
                'total_reward': total_reward,
                'epsilon': self.agent.epsilon,
                'q_table_size': len(self.agent.q_table)
            })
            
            if episode % 50 == 0:
                action_dist = self.agent.action_counts / max(np.sum(self.agent.action_counts), 1)
                print(f"Episode {episode}, Avg Reward: {np.mean(episode_rewards[-50:]):.2f}, "
                      f"Epsilon: {self.agent.epsilon:.3f}, HOLD%: {action_dist[0]*100:.1f}%")
        
        print(f"Training completed! Final epsilon: {self.agent.epsilon:.3f}")
        
        # Save the trained model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_file = f"{self.model_dir}/rl_v2g_agent_{timestamp}.pkl"
        self.agent.save_model(model_file)
        
        return episode_rewards, model_file
    
    def load_pretrained_model(self, model_path=None):
        """Load a pre-trained model"""
        if model_path is None:
            # Look for the most recent model
            model_files = [f for f in os.listdir(self.model_dir) if f.startswith("rl_v2g_agent_") and f.endswith(".pkl")]
            if not model_files:
                print("❌ No trained models found!")
                return False
            
            # Get the most recent model
            model_files.sort(reverse=True)
            model_path = os.path.join(self.model_dir, model_files[0])
        return self.agent.load_model(model_path)
    def rl_v2g_decision(self, car_session, current_slot, tariff_df):
        """Make V2G decision using trained RL agent"""
        env = V2GEnvironment(car_session, tariff_df, current_slot)
        state = env.get_state()
        
        # Use trained agent to make decision
        action = self.agent.act(state)
        result = env.step(action)
        result.update({
            'cost_revenue_inr': abs(result['energy_kwh']) * result['rate_inr_per_kwh'] if result['action'] == 'CHARGE' 
                               else abs(result['energy_kwh']) * result['rate_inr_per_kwh'] * 0.90 if result['action'] == 'DISCHARGE' 
                               else 0,
            'charging_urgency': max(0, (car_session['required_soc'] - car_session['current_soc'])) / 0.3,
            'available_v2g_energy': max(0, (car_session['current_soc'] - max(car_session['required_soc'] + 0.05, 0.20)) * car_session['battery_capacity']),
            'slots_remaining': car_session['departure_slot'] - current_slot if car_session['departure_slot'] <= 48 else (car_session['departure_slot'] - current_slot) % 48
        })
        return result
    def simulate_comprehensive_day(self, ev_sessions, tariff_df, use_pretrained=True, model_path=None):
        """Enhanced simulation with better tracking"""
        print("Running enhanced simulation with RL-based V2G agent...")
        
        model_loaded = False
        if use_pretrained:
            model_loaded = self.load_pretrained_model(model_path)
        
        if not model_loaded:
            print("🔄 No pretrained model found or failed to load. Training new model...")
            self.save_training_dataset(ev_sessions, tariff_df)
            
            episode_rewards, model_file = self.train_agent_on_historical_data(ev_sessions, tariff_df, episodes=200)
            print(f"✅ New model trained and saved: {model_file}")
        else:
            print("✅ Using pretrained model for simulation.")
        
        total_loads = []
        car_activities = []
        all_cars_all_slots = []
        
        action_stats = {'HOLD': 0, 'CHARGE': 0, 'DISCHARGE': 0}
        total_decisions = 0
        
        for slot in range(48):
            slot_time = self.slot_to_time(slot)
            
            tariff_info = tariff_df[tariff_df['slot'] == slot].iloc[0]
            rate = tariff_info['rate_inr_per_kwh']
            
            total_ev_charging_kw = 0
            total_ev_discharging_kw = 0
            total_ev_charging_kwh = 0
            total_ev_discharging_kwh = 0
            active_cars = 0
            slot_total_cost = 0
            slot_total_revenue = 0
            charging_events_this_slot = 0
            discharging_events_this_slot = 0
            hold_events_this_slot = 0
            for i, car_session in enumerate(ev_sessions):
                car_id = car_session['car_id']
                start_slot = car_session['arrival_slot']
                end_slot = car_session['departure_slot']
                is_present = False
                if end_slot > 48: 
                    is_present = slot >= start_slot or slot < (end_slot - 48)
                else:
                    is_present = start_slot <= slot < end_slot
                car_slot_data = {
                    'slot': slot,
                    'time': slot_time,
                    'car_id': car_id,
                    'group_id': car_session['group_id'],
                    'is_present': is_present,
                    'battery_capacity_kwh': car_session['battery_capacity'],
                    'required_soc': car_session['required_soc'],
                    'arrival_time': car_session['arrival_time'],
                    'departure_time': car_session['departure_time'],
                    'soc_before': round(car_session['current_soc'], 4),
                    'action': 'NOT_PRESENT',
                    'power_kw': 0,
                    'energy_kwh': 0,
                    'rate_inr_per_kwh': rate,
                    'cost_inr': 0,
                    'revenue_inr': 0,
                    'net_earning_inr': 0,
                    'cumulative_cost': round(car_session['total_costs'], 3),
                    'cumulative_revenue': round(car_session['total_earnings'], 3),
                    'soc_after': round(car_session['current_soc'], 4)
                }
                if is_present:
                    active_cars += 1
                    total_decisions += 1  
                    decision = self.rl_v2g_decision(car_session, slot, tariff_df)
                    action_stats[decision['action']] += 1
                    car_slot_data.update({
                        'action': decision['action'],
                        'power_kw': decision['power_kw'],
                        'energy_kwh': decision['energy_kwh'],
                        'charging_urgency': decision.get('charging_urgency', 0),
                        'available_v2g_energy': decision.get('available_v2g_energy', 0),
                        'slots_remaining': decision.get('slots_remaining', 0)
                    })
                    if decision['action'] == 'CHARGE':
                        cost = decision['cost_revenue_inr']
                        car_slot_data['cost_inr'] = cost
                        car_slot_data['net_earning_inr'] = -cost
                        ev_sessions[i]['total_costs'] += cost
                        slot_total_cost += cost
                        charging_events_this_slot += 1
                        soc_change = decision['energy_kwh'] / car_session['battery_capacity']
                        ev_sessions[i]['current_soc'] = min(1.0, 
                            ev_sessions[i]['current_soc'] + soc_change)
                        total_ev_charging_kw += decision['power_kw']
                        total_ev_charging_kwh += decision['energy_kwh']
                    elif decision['action'] == 'DISCHARGE':
                        revenue = decision['cost_revenue_inr']
                        car_slot_data['revenue_inr'] = revenue
                        car_slot_data['net_earning_inr'] = revenue
                        ev_sessions[i]['total_earnings'] += revenue
                        slot_total_revenue += revenue
                        discharging_events_this_slot += 1
                        soc_change = abs(decision['energy_kwh']) / car_session['battery_capacity']
                        ev_sessions[i]['current_soc'] = max(0.15, 
                            ev_sessions[i]['current_soc'] - soc_change)
                        total_ev_discharging_kw += abs(decision['power_kw'])
                        total_ev_discharging_kwh += abs(decision['energy_kwh'])
                    elif decision['action'] == 'HOLD':
                        hold_events_this_slot += 1
                    car_slot_data['cumulative_cost'] = round(ev_sessions[i]['total_costs'], 3)
                    car_slot_data['cumulative_revenue'] = round(ev_sessions[i]['total_earnings'], 3)
                    car_slot_data['soc_after'] = round(ev_sessions[i]['current_soc'], 4)
                    if decision['action'] != 'HOLD':
                        car_activities.append({
                            'slot': slot,
                            'time': slot_time,
                            'car_id': car_id,
                            'group_id': car_session['group_id'],
                            'action': decision['action'],
                            'power_kw': decision['power_kw'],
                            'energy_kwh': decision['energy_kwh'],
                            'soc_before': car_slot_data['soc_before'],
                            'soc_after': car_slot_data['soc_after'],
                            'battery_capacity_kwh': car_session['battery_capacity'],
                            'rate_inr_per_kwh': rate,
                            'cost_inr': car_slot_data['cost_inr'],
                            'revenue_inr': car_slot_data['revenue_inr'],
                            'net_earning_inr': car_slot_data['net_earning_inr'],
                            'rl_reward': decision['reward']
                        })
                all_cars_all_slots.append(car_slot_data)
            net_ev_load_kw = total_ev_charging_kw - total_ev_discharging_kw
            net_ev_energy_kwh = total_ev_charging_kwh - total_ev_discharging_kwh
            slot_net_profit = slot_total_revenue - slot_total_cost
            total_loads.append({
                'slot': slot,
                'time': slot_time,
                'ev_charging_load_kw': round(total_ev_charging_kw, 2),
                'ev_discharging_load_kw': round(total_ev_discharging_kw, 2),
                'net_ev_load_kw': round(net_ev_load_kw, 2),
                'ev_charging_energy_kwh': round(total_ev_charging_kwh, 2),
                'ev_discharging_energy_kwh': round(total_ev_discharging_kwh, 2),
                'net_ev_energy_kwh': round(net_ev_energy_kwh, 2),
                'active_cars': active_cars,
                'charging_events': charging_events_this_slot,
                'discharging_events': discharging_events_this_slot,
                'hold_events': hold_events_this_slot,
                'rate_inr_per_kwh': rate,
                'grid_demand_factor': tariff_info['grid_demand_factor'],
                'slot_total_cost_inr': round(slot_total_cost, 2),
                'slot_total_revenue_inr': round(slot_total_revenue, 2),
                'slot_net_profit_inr': round(slot_net_profit, 2)
            })
        self.hourly_loads = pd.DataFrame(total_loads)
        self.individual_car_logs = pd.DataFrame(car_activities)
        self.all_cars_all_slots = pd.DataFrame(all_cars_all_slots)
        total_charge_events = action_stats['CHARGE']
        total_v2g_events = action_stats['DISCHARGE']
        total_hold_events = action_stats['HOLD']
        hold_percentage = (total_hold_events / max(total_decisions, 1)) * 100
        v2g_participation = (total_v2g_events / max(total_charge_events, 1)) * 100
        print(f"RL Simulation Results:")
        print(f"  - Total decisions made: {total_decisions}")
        print(f"  - HOLD actions: {total_hold_events} ({hold_percentage:.1f}%)")
        print(f"  - CHARGE actions: {total_charge_events} ({total_charge_events/max(total_decisions,1)*100:.1f}%)")
        print(f"  - DISCHARGE actions: {total_v2g_events} ({total_v2g_events/max(total_decisions,1)*100:.1f}%)")
        print(f"  - V2G participation: {v2g_participation:.1f}% of charging events")       
        return self.hourly_loads, self.individual_car_logs, self.all_cars_all_slots    
    def plot_two_key_graphs(self):
        """Plot only the 2 requested graphs: Total Load vs Time and Tariff vs Time (30-min intervals)"""
        if self.hourly_loads.empty:
            print("No data to plot!")
            return
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        time_slots = self.hourly_loads['slot']
        time_labels = self.hourly_loads['time']
        time_ticks = range(0, 48, 2)  # Every hour (2 slots = 1 hour)
        ev_charging_load = self.hourly_loads['ev_charging_load_kw']
        ev_discharging_load = self.hourly_loads['ev_discharging_load_kw']
        net_load = self.hourly_loads['net_ev_load_kw']
        
        ax1.fill_between(time_slots, 0, ev_charging_load, alpha=0.7, color='green', label='EV Charging Load')
        ax1.fill_between(time_slots, 0, -ev_discharging_load, alpha=0.7, color='red', label='V2G Discharge Load')
        ax1.plot(time_slots, net_load, color='black', linewidth=3, label='Net EV Load', marker='o', markersize=2)
        ax1.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax1.set_title('Total Load vs Time (30-min intervals)', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Time (30-min slots)', fontsize=12)
        ax1.set_ylabel('Power (kW)', fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xticks(time_ticks)
        ax1.set_xticklabels([time_labels.iloc[i] for i in time_ticks], rotation=45)
        
        # 2. Tariff vs Time (30-min intervals)
        rates = self.hourly_loads['rate_inr_per_kwh']
        
        # Color code the tariff based on rate levels
        colors = []
        for rate in rates:
            if rate <= 5:
                colors.append('green')  # Low rates
            elif rate <= 8:
                colors.append('yellow')  # Medium rates
            elif rate <= 11:
                colors.append('orange')  # High rates
            else:
                colors.append('red')  # Peak rates
        
        bars = ax2.bar(time_slots, rates, color=colors, alpha=0.7, width=0.8)
        ax2.plot(time_slots, rates, color='black', linewidth=2, marker='o', markersize=2, label='Tariff Rate')
        
        ax2.set_title('Tariff vs Time (30-min intervals)', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Time (30-min slots)', fontsize=12)
        ax2.set_ylabel('Rate (₹/kWh)', fontsize=12)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_xticks(time_ticks)
        ax2.set_xticklabels([time_labels.iloc[i] for i in time_ticks], rotation=45)
        
        # Add value ranges as text
        min_rate, max_rate = rates.min(), rates.max()
        avg_rate = rates.mean()
        ax2.text(0.02, 0.98, f'Range: ₹{min_rate:.1f} - ₹{max_rate:.1f}/kWh\nAvg: ₹{avg_rate:.1f}/kWh', 
                transform=ax2.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.show()
        
        # Print summary statistics
        print("\n" + "="*60)
        print("SUMMARY STATISTICS (30-MIN INTERVALS)")
        print("="*60)
        print(f"\n📊 LOAD ANALYSIS:")
        print(f"   Peak charging load: {ev_charging_load.max():.1f} kW")
        print(f"   Peak discharge load: {ev_discharging_load.max():.1f} kW")
        print(f"   Max net load: {net_load.max():.1f} kW")
        print(f"   Min net load: {net_load.min():.1f} kW")
        
        ev_charging_energy = self.hourly_loads['ev_charging_energy_kwh']
        ev_discharging_energy = self.hourly_loads['ev_discharging_energy_kwh']
        net_energy = self.hourly_loads['net_ev_energy_kwh']
        
        print(f"\n🔋 ENERGY ANALYSIS:")
        print(f"   Total energy charged: {ev_charging_energy.sum():.1f} kWh")
        print(f"   Total energy discharged: {ev_discharging_energy.sum():.1f} kWh")
        print(f"   Net energy consumption: {net_energy.sum():.1f} kWh")
        print(f"   Energy efficiency: {(ev_discharging_energy.sum()/max(ev_charging_energy.sum(),1)*100):.1f}%")
        
        print(f"\n💰 TARIFF ANALYSIS:")
        print(f"   Tariff range: ₹{min_rate:.2f} - ₹{max_rate:.2f}/kWh")
        print(f"   Average tariff: ₹{avg_rate:.2f}/kWh")
        print(f"   Peak hours (>₹11/kWh): {len(rates[rates > 11])} slots")
        print(f"   Valley hours (<₹6/kWh): {len(rates[rates < 6])} slots")
    
    def create_ev_timeslot_matrix(self):
        """Create a matrix with EVs as rows and 30-min time slots as columns"""
        if self.all_cars_all_slots.empty:
            print("No data available to create EV-TimeSlot matrix!")
            return None
        
        # Get unique car IDs
        unique_cars = sorted(self.all_cars_all_slots['car_id'].unique())
        
        # Create time slot columns (48 slots for 24 hours with 30-min intervals)
        time_columns = []
        for slot in range(48):
            time_str = self.slot_to_time(slot)
            time_columns.append(f"Slot_{slot:02d}_{time_str}")
        
        # Initialize the matrix with basic car information
        matrix_data = []
        
        for car_id in unique_cars:
            car_data = self.all_cars_all_slots[self.all_cars_all_slots['car_id'] == car_id].iloc[0]
            
            row = {
                'Car_ID': car_id,
                'Group_ID': car_data['group_id'],
                'Battery_Capacity_kWh': car_data['battery_capacity_kwh'],
                'Required_SoC': car_data['required_soc'],
                'Arrival_Time': car_data['arrival_time'],
                'Departure_Time': car_data['departure_time'],
                'Total_Cost_INR': car_data['cumulative_cost'],
                'Total_Revenue_INR': car_data['cumulative_revenue'],
                'Net_Earning_INR': car_data['cumulative_revenue'] - car_data['cumulative_cost']
            }
            
            # Add time slot data
            car_slot_data = self.all_cars_all_slots[self.all_cars_all_slots['car_id'] == car_id]
            
            for slot in range(48):
                slot_info = car_slot_data[car_slot_data['slot'] == slot]
                
                if not slot_info.empty:
                    slot_data = slot_info.iloc[0]
                    if slot_data['is_present']:
                        # Format: Action|Power_kW|Energy_kWh|SoC_After
                        cell_value = f"{slot_data['action']}|{slot_data['power_kw']:.1f}kW|{slot_data['energy_kwh']:.2f}kWh|SoC:{slot_data['soc_after']:.2f}"
                    else:
                        cell_value = "NOT_PRESENT"
                else:
                    cell_value = "NOT_PRESENT"
                
                time_col = f"Slot_{slot:02d}_{self.slot_to_time(slot)}"
                row[time_col] = cell_value
            
            matrix_data.append(row)
        
        # Create DataFrame
        ev_matrix_df = pd.DataFrame(matrix_data)
        
        return ev_matrix_df
    
    def export_excel_files(self, filename_prefix="rl_v2g_system"):
        """Export Excel files including the new EV-TimeSlot matrix"""
        
        # Export hourly loads (main data)
        load_file = f"{filename_prefix}_hourly_loads.xlsx"
        self.hourly_loads.to_excel(load_file, index=False)
        activity_file = f"{filename_prefix}_car_activities.xlsx"
        self.individual_car_logs.to_excel(activity_file, index=False)
        complete_file = f"{filename_prefix}_all_cars_data.xlsx"
        self.all_cars_all_slots.to_excel(complete_file, index=False)
        ev_matrix = self.create_ev_timeslot_matrix()
        if ev_matrix is not None:
            matrix_file = f"{filename_prefix}_ev_timeslot_matrix.xlsx"
            with pd.ExcelWriter(matrix_file, engine='openpyxl') as writer:
                ev_matrix.to_excel(writer, sheet_name='EV_TimeSlot_Matrix', index=False)
                workbook = writer.book
                worksheet = writer.sheets['EV_TimeSlot_Matrix']
                for column in worksheet.columns:
                    max_length = 0
                    column_letter = column[0].column_letter
                    for cell in column:
                        try:
                            if len(str(cell.value)) > max_length:
                                max_length = len(str(cell.value))
                        except:
                            pass
                    adjusted_width = min(max_length + 2, 20)  # Cap at 20 characters
                    worksheet.column_dimensions[column_letter].width = adjusted_width
        print(f"\n📁 EXCEL FILES EXPORTED:")
        print(f"   📊 Hourly loads: {load_file}")
        print(f"   🚗 Car activities: {activity_file}")
        print(f"   📋 Complete data: {complete_file}")
        if ev_matrix is not None:
            print(f"   📅 EV-TimeSlot matrix: {matrix_file}")
        files = [load_file, activity_file, complete_file]
        if ev_matrix is not None:
            files.append(matrix_file)
        return files
    def list_available_models(self):
        """List all available trained models"""
        model_files = [f for f in os.listdir(self.model_dir) if f.startswith("rl_v2g_agent_") and f.endswith(".pkl")]
        if not model_files:
            print("❌ No trained models found!")
            return []
        print(f"\n📋 AVAILABLE TRAINED MODELS ({len(model_files)} found):")
        model_files.sort(reverse=True)  
        for i, model_file in enumerate(model_files):
            timestamp_str = model_file.replace("rl_v2g_agent_", "").replace(".pkl", "")
            try:
                timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                formatted_time = timestamp.strftime("%Y-%m-%d %H:%M:%S")
            except:
                formatted_time = timestamp_str
            print(f"   {i+1}. {model_file} (Trained: {formatted_time})")
        return model_files
    def get_model_info(self, model_path):
        """Get information about a specific model"""
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            print(f"\n📊 MODEL INFORMATION:")
            print(f"   📁 File: {os.path.basename(model_path)}")
            print(f"   🕒 Trained on: {model_data.get('timestamp', 'Unknown')}")
            print(f"   🧠 Q-table size: {len(model_data['q_table'])} states")
            print(f"   💾 Memory size: {len(model_data.get('memory', []))} experiences")
            print(f"   🎯 Final epsilon: {model_data['epsilon']:.4f}")
            print(f"   📈 Training episodes: {len(model_data.get('training_history', []))}")
            action_counts = model_data['action_counts']
            total_actions = np.sum(action_counts)
            if total_actions > 0:
                print(f"   📊 Action distribution:")
                actions = ['HOLD', 'CHARGE_LOW', 'CHARGE_MED', 'CHARGE_HIGH', 'DISCHARGE_LOW', 'DISCHARGE_MED', 'DISCHARGE_HIGH']
                for i, action in enumerate(actions):
                    percentage = (action_counts[i] / total_actions) * 100
                    print(f"      {action}: {percentage:.1f}%")
            return model_data
        except Exception as e:
            print(f"❌ Error reading model: {str(e)}")
            return None
def main():
    print("="*80)
    print("ENHANCED RL-BASED EV GRID MANAGEMENT WITH 30-MIN TIME SPANS")
    print("="*80)
    print("\n🎯 New Features:")
    print("   ✓ Removed Energy vs Time graph")
    print("   ✓ All graphs show 30-minute time spans")
    print("   ✓ NEW: EV-TimeSlot matrix Excel file")
    print("   ✓ 2 Key Graphs: Total Load vs Time, Tariff vs Time")
    print("   ✓ Tariff range: 3-14 INR/kWh")
    print("   ✓ Excel export with EV rows and 30-min columns")
    rl_grid_manager = RLBasedV2GGridManager()
    # List available models first
    available_models = rl_grid_manager.list_available_models()
    print("\n" + "="*50)
    print("GENERATING EV DATA AND TARIFF")
    print("="*50)
    n_cars = 100
    ev_sessions = rl_grid_manager.generate_v2g_optimized_data(n_cars)
    tariff_df = rl_grid_manager.generate_dynamic_tariff()
    print(f"\n📋 Setup:")
    print(f"   Cars: {n_cars}")
    print(f"   Time slots: 48 (30-minute intervals)")
    print(f"   Tariff range: ₹{tariff_df['rate_inr_per_kwh'].min():.1f} - ₹{tariff_df['rate_inr_per_kwh'].max():.1f}/kWh")
    print("\n" + "="*50)
    print("RUNNING RL SIMULATION")
    print("="*50)
    use_pretrained = True  # Change to False to train a new model
    total_loads_df, car_activities_df, all_cars_df = rl_grid_manager.simulate_comprehensive_day(
        ev_sessions, tariff_df, use_pretrained=use_pretrained
    )
    print("\n" + "="*50)
    print("GENERATING 2 KEY GRAPHS (30-MIN INTERVALS)")
    print("="*50)
    rl_grid_manager.plot_two_key_graphs()   
    print("\n" + "="*50)
    print("EXPORTING EXCEL FILES")
    print("="*50)   
    files = rl_grid_manager.export_excel_files("enhanced_rl_v2g_30min")   
    print("\n" + "="*50)
    print("MODEL AND DATASET INFORMATION")
    print("="*50)   
    rl_grid_manager.list_available_models()   
    if rl_grid_manager.training_dataset:
        print(f"\n📁 TRAINING DATASET SAVED:")
        print(f"   📊 Location: {os.path.dirname(rl_grid_manager.training_dataset['ev_sessions_file'])}")
        print(f"   🚗 EV Sessions: {os.path.basename(rl_grid_manager.training_dataset['ev_sessions_file'])}")
        print(f"   💰 Tariff Data: {os.path.basename(rl_grid_manager.training_dataset['tariff_file'])}")
        print(f"   📈 Cars: {rl_grid_manager.training_dataset['n_cars']}")
        print(f"   💲 Tariff Range: {rl_grid_manager.training_dataset['tariff_range']}")    
    print("\n✅ SIMULATION COMPLETE!")
    print("\n📈 Generated Graphs:")
    print("   1. Total Load vs Time (Power in kW) - 30-min intervals")
    print("   2. Tariff vs Time (Rate in ₹/kWh) - 30-min intervals")
    print("\n📁 All data exported to Excel files:")
    print("   - Standard data files")
    print("   - NEW: EV-TimeSlot Matrix (EVs as rows, 30-min slots as columns)")
    print("🤖 Trained model saved for future use")
    print("\n💡 EV-TimeSlot Matrix Format:")
    print("   Each cell: ACTION|POWER_kW|ENERGY_kWh|SoC_After")
    print("   Example: CHARGE|15.2kW|7.60kWh|SoC:0.75")
    print("\n💡 Tips:")
    print("   - Set use_pretrained=True to reuse existing models")
    print("   - Set use_pretrained=False to train a new model")
    print("   - Check 'trained_models' folder for saved models and datasets")    
    return total_loads_df, car_activities_df, all_cars_df, ev_sessions, rl_grid_manager
if __name__ == "__main__":
    total_loads, car_activities, all_cars_data, final_ev_sessions, grid_manager = main()