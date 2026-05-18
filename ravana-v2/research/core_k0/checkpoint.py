"""
RAVANA Checkpoint Manager

Crash-proof JSON saving every 500 episodes.
"""

import json
import os
import datetime


class CheckpointManager:
    """Handles periodic checkpointing and crash recovery."""
    
    def __init__(self, save_dir="checkpoints", interval=500):
        self.save_dir = save_dir
        self.interval = interval
        os.makedirs(save_dir, exist_ok=True)
        
    def save(self, episode, metrics, agent_state, env_state):
        """Save checkpoint if at interval."""
        if episode % self.interval == 0 or episode == 0:
            filename = os.path.join(self.save_dir, f"checkpoint_ep{episode}.json")
            data = {
                "episode": episode,
                "timestamp": datetime.datetime.now().isoformat(),
                "metrics": metrics,  # Dict of lists
                "agent_state": agent_state,  # Simplified state dict
                "env_state": env_state
            }
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"[CHECKPOINT] Saved episode {episode} to {filename}")
            
    def load_latest(self):
        """Find and load most recent checkpoint."""
        files = [f for f in os.listdir(self.save_dir) 
                 if f.startswith("checkpoint_ep") and f.endswith(".json")]
        if not files:
            return None
        
        # Sort by episode number
        files.sort(key=lambda x: int(x.split("ep")[1].split(".")[0]))
        latest_file = os.path.join(self.save_dir, files[-1])
        
        with open(latest_file, 'r') as f:
            data = json.load(f)
        print(f"[CHECKPOINT] Resumed from {latest_file} (Episode {data['episode']})")
        return data
