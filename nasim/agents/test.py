# -*- coding: utf-8 -*-
# @Author: shubham
# @Date:   2017-01-10 19:37:24
# @Last Modified by:   shubham
# @Last Modified time: 2017-01-10 20:23:16

import gym
from gym import wrappers

import pandas as pd
import numpy as np

import random
import itertools
from pprint import pprint
from collections import Counter

class Agent(object):
	def __init__(self, nA=None):
		self.nA = nA
	
	def set_initial_state(self, state):
		return np.random.choice(self.nA)
	
	def act(self, state, reward):
		return np.random.choice(self.nA)
		

def main():
	env = gym.make('CartPole-v0')
	outdir = './experiment-results'
	# env = wrappers.Monitor(env, directory=outdir, force=True)

	state_space = []

	agent = Agent(env.action_space.n)
	for i_episode in range(10):
		print("Episode: {}".format(i_episode+1))
		
		state = env.reset()
		action = agent.set_initial_state(state)
		for t in itertools.count():
			state_space.append(state)

			next_state, reward, done, info = env.step(action)
			action = agent.act(state, reward)
			
			env.render()
			if done:
				break
			
			state = next_state

	pprint(Counter(state_space))
	env.close()
	# gym.upload(outdir, api_key='sk_9YxUhFDaT5XSahcLut47w')


if __name__ == '__main__':
	main()
