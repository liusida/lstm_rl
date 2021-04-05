import gym
import numpy as np
from ..parameters import *

class MaskVelocityWrapper(gym.ObservationWrapper):
    """
    Gym environment observation wrapper used to mask velocity terms in
    observations. The intention is the make the MDP partially observatiable.
    """
    def __init__(self, env):
        super(MaskVelocityWrapper, self).__init__(env)
        if ENV == "CartPole-v1":
            self.mask = np.array([1., 0., 1., 0.])
        elif ENV == "Pendulum-v0":
            self.mask = np.array([1., 1., 0.])
        elif ENV == "LunarLander-v2":
            self.mask = np.array([1., 1., 0., 0., 1., 0., 1., 1,])
        elif ENV == "LunarLanderContinuous-v2":
            self.mask = np.array([1., 1., 0., 0., 1., 0., 1., 1,])
        else:
            raise NotImplementedError

    def observation(self, observation):
        return  observation * self.mask
