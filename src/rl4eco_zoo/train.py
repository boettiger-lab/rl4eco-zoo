import numpy as np
import gymnasium as gym
import stable_baselines3 as sb3

# will only import the algo that I need
# from stable_baselines3 import (
# 	A2C,
# 	DDPG,
# 	PPO,
# 	SAC,
# 	TD3
# )
# from sb3_contrib import (
# 	CrossQ,
# 	RecurrentPPO,
# 	TQC,
# 	TRPO
# )
# from sbx import (
# 	DDPG as JaxDDPG, 
# 	DQN as JaxDQN, 
# 	PPO as JaxPPO, 
# 	SAC as JaxSAC, 
# 	TD3 as JaxTD3, 
# 	TQC as JaxTQC, 
# 	CrossQ as JaxCrossQ,
# )

class WallTimeLimitCallback(BaseCallback):
    def __init__(self, max_seconds, verbose=0):
        super().__init__(verbose)
        self.max_seconds = max_seconds

    def _on_training_start(self):
        self.start_time = time.time()

    def _on_step(self):
        elapsed = time.time() - self.start_time
        if elapsed >= self.max_seconds:
            if self.verbose:
                print(f"Stopping after {elapsed:.1f} seconds.")
            return False   # Returning False stops training
        return True

def train(algo_name, env_id):
	algo = getattr(sb3, algo_name)
	env = gym.make(env_id)
	policy = (
		"MlpLstmPolicy" if algo == "RecurrentPPO" 
		else "MlpPolicy"
	)




