import gym
env = gym.make('CartPole-v0')
env.reset()
for _ in range(1000):
    env.render()
    action = env.action_space.sample()
    print(action)
    obs, reward, done, info = env.step(action)
    print(type(obs))
    print(obs, reward, done, info) # take a random action