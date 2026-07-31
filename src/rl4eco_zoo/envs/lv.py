import numpy as np
import gymnasium as gym
from gym.spaces import Box
from collections.abc import Sequence

class LotkaVolterra(gym.Env):
	"""
	Encodes a control problem on a Lotka-Volterra 
	dynamical system.
	"""
	def __init__(self, params: dict) -> None:

		########################
		# dynamical parameters #
		########################
		self.n = len(params.get('r', 1))
		self.A = params.get( # interaction matrix
			'A', 
			np.identity(self.n),
		)
		self.r = params.get( # growth rates
			'r',
			0.5 * np.ones(self.n)
		)
		#
		self.init_pops = 0.5 * np.ones(self.n)
		self.init_noise = 0.05
		self.obs_noise = 0.05
		self.bound = 2
		self.dyn_noise = 0.05
		self.ep_len = 100

		#################
		# gymnasium API #
		#################
		self.observed_species = params.get(
			'observed_species',
			[0]
		)
		self.controlled_species = params.get(
			'controlled_species',
			[0]
		)
		#
		self.observation_space = Box(
			np.array([-1] * len(self.observed_species)),
			np.array([+1] * len(self.observed_species)),
		)
		self.action_space = Box(
			np.array([-1] * len(self.controlled_species)),
			np.array([+1] * len(self.controlled_species)),
		)

		############
		# checking #
		############
		self._init_checks()

	def reset(*, seed, options):
		self.pops = (
			self.init_pops 
			+ self.init_noise * np.random.normal()
		)
		self.t = 0

	def step(action):
		#
		action = np.clip(action, -1, 1)
		expanded_act = - np.ones(self.n)
		expanded_act = np.put(
			expanded_act, 
			self.controlled_species,
			action
		)
		fraction_removed = (expanded_act + 1) / 2
		#
		reward = self.reward(self.pop, fraction_removed)
		#
		self.pops += (
			self.pops * (self.r + self.A @ self.pops)
			* (1 - fraction_removed)
		)
		self.pops = self.pops * np.random.normal(1,self.dyn_noise)
		observation = self.observe()
		#
		info = {}
		self.t+=1
		done = self.t > self.ep_len
		#
		return observation, reward, done, done, info

	def _init_checks(self):
		assert (
			isinstance(self.observed_species, Sequence),
			"LotkaVolterra: 'observed_species' parameter must " 
			"be a sequence."
		)
		assert (
			isinstance(self.controlled_species, Sequence),
			"LotkaVolterra: 'controlled_species' parameter must " 
			"be a sequence."
		)
		assert all(self.n == dim for dim in self.A.shape)


	def get_state(self):
		return 2 * self.pops / self.bound - 1

	def reward(self, pop, fraction_removed):
		return np.sum(fraction_removed * pop)

	def observe(self):
		state = self.get_state()
		full_pseudo_obs = (
			state + 
			self.obs_noise * np.random.normal()
		)
		return full_pseudo_obs[self.observed_species]

