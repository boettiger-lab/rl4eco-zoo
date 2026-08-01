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
			np.ones(self.n)
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

	def reset(*, seed=42, options=None):
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



class LVSampler:
	def __init__(self, params):
		self.sampler_type = params.get("sampler_type")
		self.A_scale = params.get("A_scale", 1)

		"""
		Note: the growth rates r simply set a time-scale for each
		species. We will assume that they all share the same scale,
		which is set to 1 for convenience.
		"""

		self._init_checks()

	def sample(self, n_sp):
		sampler_fn = getattr(self.sampler_type)
		return sampler_fn(n_sp)

	def iid_uniform(self, n_sp):
		# prefactor np.sqrt(12) is such that the 
		# variance is sdev^2 / n_sp
		return (
			self.A_scale * 
			np.ones(n_sp, n_sp) /
			np.sqrt(n_sp)
			+ 
			(
				self.A_scale *
				np.sqrt(12) *
				(np.random.rand(n_sp, n_sp) - 1/2) /
				np.sqrt(n_sp)
			)
		)

	def iid_gaussian(self, n_sp):
		return np.random.normal(
			loc = np.ones(n_sp, n_sp) / np.sqrt(n_sp),
			scale = self.A_scale / np.sqrt(n_sp)
		)

	def gaussian_symm(self, n_sp):
		A = np.random.normal(
			loc = np.ones(n_sp, n_sp) / np.sqrt(n_sp),
			scale = self.A_scale / np.sqrt(n_sp)
		)
		A = (A + A.T) / np.sqrt(2) 
		# ^sqrt instead of 2 for the correct variance in off diagonal
		np.fill_diagonal(A, A.diagonal() * np.sqrt(2))
		# ^correct diagonal
		return A


	def _init_checks(self):
		assert (
			hasattr(self, self.sampler_type) and
			callable(getattr(self,getattr(self.sampler_type)))
		), (
			"LVSampler: 'sampler_type' does not match any "
			"implemented sampler."
		)

class RandomizedLotkaVolterra(LotkaVolterra):
	def __init__(self, params):
		super().__init__(params)
		self.sampler_type = params.get("sampler_type", "uniform")
		self.A_scale = params.get("A_scale", 1)
		self.sampler = LVSampler(
			params = {
				'A_scale': self.A_scale,
				'sampler_type': self.sampler_type
			}
		)

	def reset(*, seed=42, options=None):
		super().reset(seed=seed,options=options)
		self.A = self.sampler.sample(n_sp = self.n)


