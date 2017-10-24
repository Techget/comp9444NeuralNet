import gym
import sys

env = gym.make('MountainCar-v0')
env.reset()
reward = -1.0
# for _ in range(1000):
while True:
	# state = env.reset()
	env.render()
	ep_max_steps = 200
	for step in range(ep_max_steps):
	    # env.render()
	    action = env.action_space.sample()
	    # print(action)
	    obs, reward, done, info = env.step(action)
	    print(obs, reward, done, info) # take a random action
	    if obs[0] >= 0.5:
	    	sys.exit()

	    # print(type(obs))
	    