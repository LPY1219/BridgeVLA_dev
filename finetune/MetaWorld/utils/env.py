import torch
import gym
import numpy as np
import matplotlib.pyplot as plt
import os
import metaworld
import random
import time

from natsort import natsorted
from termcolor import cprint
from gym import spaces

from utils.camera import Camera




class MetaWorldEnv(gym.Env):
    metadata = {"render.modes": ["rgb_array"], "video.frames_per_second": 10}

    def __init__(self, task_name, device="cuda:0"):
        super(MetaWorldEnv, self).__init__()

        if '-v2' not in task_name:
            task_name = task_name + '-v2-goal-observable'

        self.instruction = task_name.replace('-', ' ')

        self.env = metaworld.envs.ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[task_name]()
        self.env._freeze_rand_vec = False

        # https://arxiv.org/abs/2212.05698
        # self.env.sim.model.cam_pos[2] = [0.75, 0.075, 0.7]
        self.env.sim.model.cam_pos[2] = [0.6, 0.295, 0.8]
        

        self.env.sim.model.vis.map.znear = 0.1
        self.env.sim.model.vis.map.zfar = 1.5
        
        self.device_id = int(device.split(":")[-1])
        
        self.image_size = 128
        
        # Initialize camera after reset to avoid segmentation fault
        # Try to use all cameras, but Camera class will filter out non-existent ones
        try:
            self.camera = Camera(sim=self.env.sim, cam_names=['topview', 'corner', 'corner2', 'corner3', 'behindGripper', 'gripperPOV'], img_size=self.image_size)
        except Exception as e:
            # Fallback to single camera if multiple cameras fail
            print(f"Warning: Failed to initialize multiple cameras: {e}")
            print("Falling back to single camera 'corner2'")
            self.camera = Camera(sim=self.env.sim, cam_names=['corner2'], img_size=self.image_size)
    
        self.episode_length = self._max_episode_steps = 410
        self.action_space = self.env.action_space
        self.obs_sensor_dim = self.get_robot_state().shape[0]

        self.num_points = self.image_size * self.image_size * len(self.camera.cam_names)

        self.observation_space = spaces.Dict({
            'agent_pos': spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.obs_sensor_dim,),
                dtype=np.float32
            ),
            'point_cloud': spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.num_points, 3),
                dtype=np.float32
            ),
            'full_state': spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(20, ),
                dtype=np.float32
            ),
        })

    def get_robot_state(self):
        eef_pos = self.env.get_endeff_pos()
        finger_right, finger_left = (
            self.env._get_site_pos('rightEndEffector'),
            self.env._get_site_pos('leftEndEffector')
        )
        return np.concatenate([eef_pos, finger_right, finger_left])

    def get_rgb(self):
        # cam names: ('topview', 'corner', 'corner2', 'corner3', 'behindGripper', 'gripperPOV')
        img = self.env.sim.render(width=self.image_size, height=self.image_size, camera_name="corner2", device_id=self.device_id)
        return img

    def render_high_res(self, resolution=1024):
        img = self.env.sim.render(width=resolution, height=resolution, camera_name="corner2", device_id=self.device_id)
        return img
    

    def get_point_cloud(self, use_rgb=True):
        point_cloud, depth = self.camera.capturePointCloud(device_id=self.device_id) # raw point cloud, Nx3

        if not use_rgb:
            point_cloud = point_cloud[..., :3]

        depth = depth[::-1]
        
        return point_cloud, depth
        

    def get_visual_obs(self):
        obs_pixels = self.get_rgb()
        robot_state = self.get_robot_state()
        point_cloud, depth = self.get_point_cloud()
        
        if obs_pixels.shape[0] != 3:
            obs_pixels = obs_pixels.transpose(2, 0, 1)

        obs_dict = {
            'image': obs_pixels,
            'depth': depth,
            'agent_pos': robot_state,
            'point_cloud': point_cloud,
            'instruction': self.instruction,
        }
        return obs_dict
            
            
    def step(self, action: np.array):
        raw_state, reward, done, env_info = self.env.step(action)
        self.cur_step += 1

        obs_pixels = self.get_rgb()
        robot_state = self.get_robot_state()
        point_cloud, depth = self.get_point_cloud()
        
        if obs_pixels.shape[0] != 3:  # make channel first
            obs_pixels = obs_pixels.transpose(2, 0, 1)

        obs_dict = {
            'image': obs_pixels,
            'depth': depth,
            'agent_pos': robot_state,
            'point_cloud': point_cloud,
            'full_state': raw_state,
        }

        done = done or self.cur_step >= self.episode_length
        return obs_dict, reward, done, env_info

    def reset(self):
        self.env.reset()
        self.env.reset_model()
        
        raw_obs = self.env.reset()
        self.cur_step = 0

        obs_pixels = self.get_rgb()
        robot_state = self.get_robot_state()
        point_cloud, depth = self.get_point_cloud()
        
        if obs_pixels.shape[0] != 3:
            obs_pixels = obs_pixels.transpose(2, 0, 1)

        obs_dict = {
            'image': obs_pixels,
            'depth': depth,
            'agent_pos': robot_state,
            'point_cloud': point_cloud,
            'full_state': raw_obs,
        }


        return obs_dict

    def seed(self, seed=None):
        pass

    def set_seed(self, seed=None):
        pass

    def render(self, mode='rgb_array'):
        img = self.get_rgb()
        return img

    def close(self):
        pass

