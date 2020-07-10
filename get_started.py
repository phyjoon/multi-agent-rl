import time
import numpy as np

import flatland
from flatland.envs.rail_env import RailEnv
from flatland.utils.rendertools import RenderTool
from flatland.envs.rail_generators import complex_rail_generator
from flatland.envs.schedule_generators import complex_schedule_generator


if __name__ == "__main__":
    env = RailEnv(width=50, height=50,
                  number_of_agents=1) 

    obs = env.reset()
    obs, all_rewards, done, _ = env.step({0: 0})

    env_renderer = RenderTool(env)
    env_renderer.render_env(show=True, frames=True, show_observations=False)
    
    for step in range(100):
        action = np.random.randint(0, 5) #np.argmax(obs[0])+1
        obs, all_rewards, done, _ = env.step({0:action})
        print("Rewards: ", all_rewards, "  [done=", done, "]")

        env_renderer.render_env(show=True, frames=True, show_observations=False)
        time.sleep(0.1)
        