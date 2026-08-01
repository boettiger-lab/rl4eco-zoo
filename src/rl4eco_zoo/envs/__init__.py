import gymnasium as gym

gym.register(
	id = "LV-v0",
	entry_point = "rl4eco_zoo.envs:LotkaVolterra",
)

gym.register(
	id = "RandLV-v0",
	entry_point = "rl4eco_zoo.envs:RandomizedLotkaVolterra",
)