import gym
import random

env_name='CartPole-v1'
env = gym.make(env_name)
MAX_STEPS = 500
MAX_ANGLE = 45
env.reset()

print('Number of Obs:', env.observation_space)
print('Number of actions:', env.action_space)

class Agent():
    def __init__(self, env=env):
        self.action_size = env.action_space.n
        print('Action size:', self.action_size)

    def get_rand_action(self):
        action = random.randrange(0, self.action_size)
        return action

    def get_logical_action(self, obs):
        pole_angle = obs[2]
        print(pole_angle)
        action = 0 if pole_angle<0 else 1
        return action


agent = Agent(env)
obs = env.reset()
env._max_episode_steps = MAX_STEPS
env.set_max_angle(MAX_ANGLE)
score = 0
for _ in range(MAX_STEPS):
    # action = agent.get_rand_action()
    action = agent.get_logical_action(obs)
    # print(action)
    obs, reward, done, info = env.step(action)
    env.render()
    score+=reward
    if done:
        break

print('Score:', score)
