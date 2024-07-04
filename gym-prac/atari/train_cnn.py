from helpful import *

env_name = ""

env = gym.make(env_name, render_mode = 'rgb_array')
env.metadata['render_fps'] = 30

# Setup the wrapper to record the video
env = RecordVideo(env, video_folder='./gym-prac/videos', episode_trigger=lambda episode_id: (episode_id % 50) == 0)

def train():
    pass