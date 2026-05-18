import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os

def generate_synthetic_student_data(n_records=10000, output_path="results/synthetic_student_interactions.csv"):
    np.random.seed(42)
    
    # 1. Setup Demographics (Imbalanced for Fairness Testing)
    # Group A: 50% (High success baseline)
    # Group B: 30% (Lower success baseline)
    # Group C: 20% (Medium success baseline)
    groups = ['Group A', 'Group B', 'Group C']
    group_probs = [0.5, 0.3, 0.2]
    
    # 2. Interaction Types
    interaction_types = ['MCQ', 'Open-ended', 'Problem-solving']
    
    data = []
    start_time = datetime(2026, 1, 15)
    
    print(f"Generating {n_records} synthetic student records...")
    
    for i in range(n_records):
        group = np.random.choice(groups, p=group_probs)
        interaction = np.random.choice(interaction_types)
        
        # Base success probability (Simulated bias)
        if group == 'Group A':
            base_success = 0.80
        elif group == 'Group B':
            base_success = 0.60 # The 20% gap to fix
        else:
            base_success = 0.70
            
        # Add noise based on interaction type
        if interaction == 'Open-ended':
            noise = np.random.normal(0, 0.15)
        else:
            noise = np.random.normal(0, 0.05)
            
        response_quality = np.clip(base_success + noise, 0, 1)
        
        # Adversarial flags (5%)
        is_adversarial = np.random.random() < 0.05
        
        # Temporal drift (slightly declining quality over the semester to test adaptation)
        days_offset = (i / n_records) * 120 # 4 months
        timestamp = start_time + timedelta(days=days_offset)
        
        # Metadata for dissonance
        noise_level = abs(noise)
        
        data.append({
            'student_id': f"STU_{i:05d}",
            'demographic_group': group,
            'interaction_type': interaction,
            'response_quality': float(response_quality),
            'timestamp': timestamp.isoformat(),
            'adversarial_flag': int(is_adversarial),
            'noise_level': float(noise_level),
            'base_success_rate': base_success # Ground truth for validation
        })
        
    df = pd.DataFrame(data)
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    df.to_csv(output_path, index=False)
    print(f"Successfully saved synthetic data to {output_path}")
    
    # Print summary statistics for verification
    print("\nInitial Demographic Parity Gap Check (Raw Data):")
    summary = df.groupby('demographic_group')['response_quality'].mean()
    print(summary)
    gap_a_b = summary['Group A'] - summary['Group B']
    print(f"Gap A-B: {gap_a_b:.2%}")
    
    return output_path

if __name__ == "__main__":
    generate_synthetic_student_data()
