import gymnasium as gym
from gymnasium.wrappers import RecordVideo

from REINFORCE import REINFORCE
from AC import ActorCritic
from helpful import *


env_name = "CartPole-v1" #"InvertedDoublePendulum-v4" # "Humanoid-v4" # "BipedalWalker-v3" #  LunarLanderContinuous-v2  "BipedalWalker-v3" Hardcore continous
env_info = query_env(env_name)

env = gym.make(env_name, render_mode = 'rgb_array')
env.metadata['render_fps'] = 30

# Setup the wrapper to record the video
env = RecordVideo(env, video_folder='./videos', episode_trigger=lambda episode_id: (episode_id % 50) == 0)

agent = REINFORCE(env_info)

episodeNumper = 3000
for episode_index in range(episodeNumper):
    observation, _ = env.reset()
    done = False
    while not done:
        action = agent.sample_action(observation)
        if env_info['action_space_continuous']:
            observation, reward, done, truncated, info = env.step(action)
        else:
            observation, reward, done, truncated, info = env.step(int(action))
        agent.rewards.append(reward)
        done = done or truncated

    if (episode_index) % 100 == 0:
        print(f"Episode {episode_index + 1} : " , "Reward : ", int(np.sum(np.array(agent.rewards))))

    agent.update()

env.close