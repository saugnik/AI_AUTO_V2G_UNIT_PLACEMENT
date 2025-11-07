import numpy as np
import pandas as pd
import pickle
import os

class V2GInference:
    def __init__(self, model_path):
        """Load trained model for inference"""
        with open(model_path, 'rb') as f:
            self.model_data = pickle.load(f)
        self.q_table = self.model_data['q_table']
        print(f"Model loaded: {len(self.q_table)} states")
    
    def discretize_state(self, state):
        """Convert state to discrete for Q-table lookup"""
        return tuple([round(s, 1) for s in state])
    
    def predict(self, state):
        """Predict action for given state"""
        discrete_state = self.discretize_state(state)
        if discrete_state in self.q_table:
            return np.argmax(self.q_table[discrete_state])
        return 0 

def get_state(car_soc, required_soc, battery_capacity, rate, grid_demand, slots_remaining, current_slot):
    """Create state vector"""
    energy_needed = max(0, (required_soc - car_soc) * battery_capacity)
    available_v2g = max(0, (car_soc - max(required_soc + 0.05, 0.20)) * battery_capacity)
    
    return np.array([
        car_soc,
        required_soc,
        energy_needed / battery_capacity,
        available_v2g / battery_capacity,
        rate / 14.0,
        grid_demand,
        slots_remaining / 48.0,
        (current_slot % 48) / 48.0,
        battery_capacity / 100.0,
        min(1.0, energy_needed / (slots_remaining * 5.0 + 0.1)),
        min(1.0, available_v2g / 10.0)
    ])

def run_inference(model_path, car_data, tariff_data):
    """Run inference on car and tariff data"""
    # Load model
    model = V2GInference(model_path)
    
    actions = ['HOLD', 'CHARGE_LOW', 'CHARGE_MED', 'CHARGE_HIGH', 
               'DISCHARGE_LOW', 'DISCHARGE_MED', 'DISCHARGE_HIGH']
    
    results = []
    
    for slot in range(48):
        tariff = tariff_data[tariff_data['slot'] == slot].iloc[0]
        rate = tariff['rate_inr_per_kwh']
        grid_demand = tariff['grid_demand_factor']
        
        for car in car_data:
            # Check if car is present
            if car['arrival_slot'] <= slot < car['departure_slot']:
                slots_remaining = car['departure_slot'] - slot
                
                # Get state
                state = get_state(
                    car['current_soc'], 
                    car['required_soc'],
                    car['battery_capacity'],
                    rate,
                    grid_demand,
                    slots_remaining,
                    slot
                )
                
                # Predict action
                action_idx = model.predict(state)
                action_name = actions[action_idx]
                
                results.append({
                    'slot': slot,
                    'car_id': car['car_id'],
                    'action': action_name,
                    'action_index': action_idx,
                    'soc': car['current_soc'],
                    'rate': rate,
                    'grid_demand': grid_demand
                })
    
    return pd.DataFrame(results)

if __name__ == "__main__":
    car_data = []
for i in range(25):
    car_data.append({
        'car_id': i + 1,
        'arrival_slot': np.random.randint(0, 24),              # random arrival
        'departure_slot': np.random.randint(24, 48),           # random departure after arrival
        'current_soc': np.random.uniform(0.2, 0.8),
        'required_soc': np.random.uniform(0.6, 0.95),
        'battery_capacity': np.random.choice([40, 50, 60, 70])
    })
    
    # Example tariff data
    tariff_data = pd.DataFrame({
        'slot': range(48),
        'rate_inr_per_kwh': np.random.uniform(3, 14, 48),
        'grid_demand_factor': np.random.uniform(0.1, 1.0, 48)
    })
    
    model_path = r"trained_models\rl_v2g_agent_20250804_014401.pkl"
    
    print(f"Checking model path: {model_path}")
    print(f"Path exists: {os.path.exists(model_path)}")
    
    try:
        if os.path.exists(model_path):
            print("Loading model...")
            results = run_inference(model_path, car_data, tariff_data)
            print("Inference completed!")
            print(results.head(100))  # Show only 100 rows
            print(f"\nAction distribution:")
            print(results['action'].value_counts())
        else:
            print(f"Model file not found: {model_path}")
            print("Current directory:", os.getcwd())
            print("Files in 'new' directory:")
            if os.path.exists("new"):
                for root, dirs, files in os.walk("new"):
                    for file in files:
                        if file.endswith('.pkl'):
                            print(f"  Found: {os.path.join(root, file)}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
