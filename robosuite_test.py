from secant.envs.robosuite import make_robosuite
from PIL import Image

env = make_robosuite(
    task="Door",
    mode="eval-easy",
    scene_id=2,
    image_width=600,
    image_height=600,
)
env.reset()

done = False
while not done:
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    print("reward = ", reward)
    env.render()
