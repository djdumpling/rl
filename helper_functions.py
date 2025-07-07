import torch as t
import random
import numpy as np
import gymnasium as gym
import os

def set_global_seeds(seed):
    t.manual_seed(seed)
    t.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    t.backends.cudnn.deterministic = True

def make_env(env_id, seed, idx, run_name, mode = "classic-control", video_log_freq = None, video_save_path = None):
    def thunk():
        env = gym.make(env_id, render_mode = "rgb_array")
        env = gym.wrappers.RecordEpisodeStatistics(env)
        
        if idx == 0 and video_log_freq:
            # Create the full video directory path for this run
            video_dir = f"{video_save_path}/{run_name}"
            os.makedirs(video_dir, exist_ok=True)
            
            env = gym.wrappers.RecordVideo(
                env, 
                video_dir,
                episode_trigger = lambda episode_id: episode_id % video_log_freq == 0,
                disable_logger = True,
                video_format = "mp4",
                force_fps = 30
            )

        env.reset(seed = seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)

        return env
    
    return thunk

def get_episode_data_from_infos(infos: dict) -> dict[str, int | float] | None:
    """
    Helper function: returns dict of data from the first terminated environment, if at least one terminated.
    """
    for final_info in infos.get("final_info", []):
        if final_info is not None and "episode" in final_info:
            return {
                "episode_length": final_info["episode"]["l"].item(),
                "episode_reward": final_info["episode"]["r"].item(),
                "episode_duration": final_info["episode"]["t"].item(),
            }