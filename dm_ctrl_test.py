from secant.envs.dm_control import make_dmc

env = make_dmc(task="walker_walk")
env.reset()

done = False
while not done:
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    env.render()
