import wandb
from typing import Dict, Callable
import numpy as np
import torch
import collections
import tqdm
import os
import json
from utils.env import MetaWorldEnv
from utils.multistep_wrapper import MultiStepWrapper
from utils.video_recording_wrapper import SimpleVideoRecordingWrapper
from utils.camera import get_grasp
import mujoco
import cv2

from termcolor import cprint
import heapq

# Try to import imageio for video saving, fallback to cv2
try:
    import imageio
    HAS_IMAGEIO = True
except ImportError:
    HAS_IMAGEIO = False
    try:
        import cv2
        HAS_CV2 = True
    except ImportError:
        HAS_CV2 = False

def dict_apply(
        x: Dict[str, torch.Tensor], 
        func: Callable[[torch.Tensor], torch.Tensor]
        ) -> Dict[str, torch.Tensor]:
    result = dict()
    for key, value in x.items():
        if isinstance(value, dict):
            result[key] = dict_apply(value, func)
        else:
            result[key] = func(value)
    return result

class LargestKRecorder:
    def __init__(self, K):
        """
        Initialize the EfficientScalarRecorder.
        
        Parameters:
        - K: Number of largest scalars to consider when computing the average.
        """
        self.scalars = []
        self.K = K

    def record(self, scalar):
        """
        Record a scalar value.
        
        Parameters:
        - scalar: The scalar value to be recorded.
        """
        if len(self.scalars) < self.K:
            heapq.heappush(self.scalars, scalar)
        else:
            # Compare the new scalar with the smallest value in the heap
            if scalar > self.scalars[0]:
                heapq.heappushpop(self.scalars, scalar)

    def average_of_largest_K(self):
        """
        Compute the average of the largest K scalar values recorded.
        
        Returns:
        - avg: Average of the largest K scalars.
        """
        if len(self.scalars) == 0:
            raise ValueError("No scalars have been recorded yet.")
        
        return sum(self.scalars) / len(self.scalars)
    
class MetaworldRunner:
    def __init__(self,
                 output_dir,
                 eval_episodes=20,
                 max_steps=2000,
                 n_obs_steps=10,
                 n_action_steps=10,
                 fps=10,
                 crf=22,
                 tqdm_interval_sec=5.0,
                 task_name=None,
                 device="cuda:0",
            ):
        self.task_name = task_name
        self.device = device
        self.output_dir = output_dir

        def env_fn(task_name):
            return MultiStepWrapper(
                SimpleVideoRecordingWrapper(
                    MetaWorldEnv(task_name=task_name,device=device)),
                n_obs_steps=n_obs_steps,
                n_action_steps=n_action_steps,
                max_episode_steps=max_steps,
                reward_agg_method='sum',
            )
        
        self.eval_episodes = eval_episodes
        self.env = env_fn(self.task_name)
        self.instruction = self.task_name.replace('-', ' ')

        self.fps = fps
        self.crf = crf
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.max_steps = max_steps
        self.tqdm_interval_sec = tqdm_interval_sec

        self.logger_util_test = LargestKRecorder(K=3)
        self.logger_util_test10 = LargestKRecorder(K=5)

    def _save_video(self, videos, episode_idx, video_dir):
        """
        Save video to local file system.
        
        Args:
            videos: Video array with shape (T, C, H, W)
            episode_idx: Episode index for filename
            video_dir: Directory to save video
        """
        # videos shape is (T, C, H, W), need to convert to (T, H, W, C) for saving
        if len(videos.shape) == 5:
            videos = videos[:, 0]  # select first frame
        
        video_for_save = videos.transpose(0, 2, 3, 1)  # (T, C, H, W) -> (T, H, W, C)
        
        # Ensure video is uint8
        if video_for_save.dtype != np.uint8:
            video_for_save = np.clip(video_for_save, 0, 255).astype(np.uint8)
        
        # Generate video filename
        video_filename = f"eval_{self.task_name}_episode_{episode_idx:03d}.mp4"
        video_path = os.path.join(video_dir, video_filename)
        
        # Save video using imageio or cv2
        try:
            if HAS_IMAGEIO:
                imageio.mimsave(video_path, video_for_save, fps=self.fps, codec='libx264', quality=8)
                cprint(f"Video saved to: {video_path}", 'green')
            elif HAS_CV2:
                # Use cv2 to save video
                H, W = video_for_save.shape[1:3]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(video_path, fourcc, self.fps, (W, H))
                for frame in video_for_save:
                    # Convert RGB to BGR for OpenCV
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    out.write(frame_bgr)
                out.release()
                cprint(f"Video saved to: {video_path}", 'green')
            else:
                cprint("Warning: Neither imageio nor cv2 available. Video not saved locally.", 'yellow')
                return None
        except Exception as e:
            cprint(f"Error saving video to {video_path}: {e}", 'red')
            return None
        
        return video_path

    def run(self, policy, save_video=False):
        all_traj_rewards = []
        all_success_rates = []
        all_total_steps = []
        env = self.env
        
        # Create video directory if saving videos
        video_dir = None
        if save_video:
            video_dir = os.path.join(self.output_dir, "videos")
            os.makedirs(video_dir, exist_ok=True)
            cprint(f"Videos will be saved to: {video_dir}", 'cyan')

        last_videos = None  # Store last video for wandb
        
        for episode_idx in tqdm.tqdm(range(self.eval_episodes), desc=f"Eval in Metaworld {self.task_name} Pointcloud Env", leave=False, mininterval=self.tqdm_interval_sec):
            
            # start rollout
            obs = env.reset()

            done = False
            info = {'success': False}
            traj_reward = 0
            is_success = False
            total_steps = 0
            while True:
                if done or bool(info['success']):
                    break
                print("**********, obj_to_target: ", info['obj_to_target'])
                np_obs_dict = dict[bytes, bytes](obs)
                obs_dict = dict_apply(np_obs_dict,
                                      lambda x: torch.from_numpy(np.ascontiguousarray(x)).to(
                                          device=self.device))

                with torch.no_grad():
                    obs_dict_input = {}
                    obs_dict_input['point_cloud'] = obs_dict['point_cloud']
                    obs_dict_input['agent_pos'] = obs_dict['agent_pos']
                    obs_dict_input['instruction'] = self.instruction
                    action = policy.predict_action(obs_dict_input)
                    total_steps += action.shape[0]

                obs, reward, done, info = env.step(action)

                traj_reward += reward
                done = np.all(done)
                is_success = is_success or bool(info['success'])
            
            print(f"\n total_steps: {total_steps}")
            print(f"\n Is success: {is_success}")

            all_success_rates.append(is_success)
            all_traj_rewards.append(traj_reward)
            all_total_steps.append(total_steps)
            # Save video for each episode
            if save_video:
                videos = env.env.get_video()
                if videos is not None and len(videos) > 0:
                    # Save to local file system
                    self._save_video(videos, episode_idx, video_dir)
                    # Store last video for wandb
                    last_videos = videos.copy() if hasattr(videos, 'copy') else videos
            
            # Reset environment for next episode
            _ = env.reset()

        max_rewards = collections.defaultdict(list)
        log_data = dict()

        log_data['mean_traj_rewards'] = np.mean(all_traj_rewards)
        log_data['mean_success_rates'] = np.mean(all_success_rates)

        log_data['test_mean_score'] = np.mean(all_success_rates)
        
        cprint(f"test_mean_score: {np.mean(all_success_rates)}", 'green')
        cprint(f"all_total_steps: {np.mean(all_total_steps)}", 'green')

        self.logger_util_test.record(np.mean(all_success_rates))
        self.logger_util_test10.record(np.mean(all_success_rates))
        log_data['SR_test_L3'] = self.logger_util_test.average_of_largest_K()
        log_data['SR_test_L5'] = self.logger_util_test10.average_of_largest_K()
        
        # Create wandb video object from last episode (if available)
        if save_video and last_videos is not None:
            if len(last_videos.shape) == 5:
                last_videos = last_videos[:, 0]  # select first frame
            
            videos_wandb = wandb.Video(last_videos, fps=self.fps, format="mp4")
            log_data[f'sim_video_eval'] = videos_wandb
            
            # Print wandb storage location if available
            if wandb.run:
                wandb_video_dir = os.path.join(wandb.run.dir, "files", "media")
                cprint(f"WandB video will be stored in: {wandb_video_dir}", 'cyan')
            else:
                cprint("WandB not initialized. Video will be uploaded when wandb.log() is called.", 'yellow')

        # Save data to JSON file
        json_data = {
            'task_name': self.task_name,
            'eval_episodes': self.eval_episodes,
            'episode_data': [
                {
                    'episode_idx': idx,
                    'success_rate': bool(success),  # Convert numpy bool to Python bool
                    'traj_reward': float(reward),   # Convert numpy float to Python float
                    'total_steps': int(steps)       # Convert numpy int to Python int
                }
                for idx, (success, reward, steps) in enumerate(zip(all_success_rates, all_traj_rewards, all_total_steps))
            ],
            'summary_statistics': {
                'mean_traj_rewards': float(log_data['mean_traj_rewards']),
                'mean_success_rates': float(log_data['mean_success_rates']),
                'test_mean_score': float(log_data['test_mean_score']),
                'SR_test_L3': float(log_data['SR_test_L3']),
                'SR_test_L5': float(log_data['SR_test_L5']),
                'mean_total_steps': float(np.mean(all_total_steps))
            }
        }
        
        # Save JSON file
        json_filename = f"eval_results_{self.task_name}.json"
        json_path = os.path.join(self.output_dir, json_filename)
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        cprint(f"Evaluation results saved to: {json_path}", 'green')

        return log_data


class MetaworldRunnerGrasping(MetaworldRunner):
    def __init__(self,
                 output_dir,
                 eval_episodes=20,
                 max_steps=2000,
                 n_obs_steps=10,
                 n_action_steps=10,
                 fps=10,
                 crf=22,
                 tqdm_interval_sec=5.0,
                 task_name=None,
                 device="cuda:0",
                 ):
        super().__init__(output_dir, eval_episodes, max_steps, n_obs_steps, n_action_steps, fps, crf, tqdm_interval_sec, task_name, device)
        self.device_id = int(device.split(":")[-1])
        
        # Get the directory where this file is located
        utils_dir = os.path.dirname(__file__)
        grasping_dir = os.path.join(utils_dir, "grasping")
        with open(os.path.join(grasping_dir, "name2maskid.json"), "r") as f:
            name2maskid = json.load(f)
        with open(os.path.join(grasping_dir, "name2mode.json"), "r") as f:
            name2mode = json.load(f)
        with open(os.path.join(grasping_dir, "name2graspdev.json"), "r") as f:
            name2graspdev = json.load(f)
            
        env_name = self.task_name + '-v2-goal-observable'
        self.mode = name2mode[env_name]
        self.seg_ids = name2maskid[env_name]
        self.grasp_dev = name2graspdev[env_name]
        
    def get_grasp_position(self):
        grasping_calc_cam = "corner"
        depth = self.env.camera.captureDepth(grasping_calc_cam, device_id=self.device_id)
        seg_img = self.env.camera.captureSeg(grasping_calc_cam, device_id=self.device_id)
        sed_obj = self.get_seg(seg_img, seg_ids=self.seg_ids)
        cv2.imwrite(f"/DATA/disk1/cyx/BridgeVLA_dev/finetune/MetaWorld/seg_img_{self.task_name}.png", sed_obj)
        

        cam_w2c = self.env.camera.get_camera_transform_matrix(self.env.camera.env_all_cameras.index(grasping_calc_cam))
        self.grasp = get_grasp(sed_obj, depth, np.linalg.inv(cam_w2c))
        self.init_grasp()
        self.grasped = False
        
    def init_grasp(self):
        self.grasped = False
        self.grasp[:2] = self.grasp[:2] + np.array(self.grasp_dev)
        if self.mode == "push":
            self.grasp = self.grasp - np.array([0, 0, 0.06])

    
    def get_seg(self, seg_img, seg_ids):
        img = np.zeros(seg_img.shape[:2], dtype=bool)
        types = seg_img[:, :, 0]
        ids = seg_img[:, :, 1]
        geoms = types == mujoco.mjtObj.mjOBJ_GEOM
        geoms_ids = np.unique(ids[geoms])

        for i in geoms_ids:
            if (i+4) in seg_ids:
                img[ids == i] = True
        img = img.astype('uint8') * 255
        return cv2.medianBlur(img, 3)
    
    def get_action_heuristic(self, obs_dict):
        curr_pose = obs_dict['agent_pos'][:3].cpu().numpy()
        if not self.grasped and np.linalg.norm(curr_pose[:2] - self.grasp[:2]) > 0.02:
            return self.grasp + np.array([0, 0, 0.2])
        elif not self.grasped and np.linalg.norm(curr_pose[2] - self.grasp[2]) > 0.04:
            return self.grasp
        else:
            self.grasped = True
            return self.grasp
        
    def get_grab_effort(self, obs_dict):
        curr_pose = obs_dict['agent_pos'][:3].cpu().numpy()
        if self.grasped or self.mode == "push" or (not self.grasped and np.linalg.norm(curr_pose[2] - self.grasp[2]) < 0.08):
            return 0.8
        else:
            return -0.8
        
    
    def run(self, policy, save_video=False):
        all_traj_rewards = []
        all_success_rates = []
        all_total_steps = []
        env = self.env
        
        # Create video directory if saving videos
        video_dir = None
        if save_video:
            video_dir = os.path.join(self.output_dir, "videos")
            os.makedirs(video_dir, exist_ok=True)
            cprint(f"Videos will be saved to: {video_dir}", 'cyan')

        last_videos = None  # Store last video for wandb
        
        for episode_idx in tqdm.tqdm(range(self.eval_episodes), desc=f"Eval in Metaworld {self.task_name} Pointcloud Env", leave=False, mininterval=self.tqdm_interval_sec):
            
            # start rollout
            obs = env.reset()
            self.get_grasp_position()
            
            np_obs_dict = dict[bytes, bytes](obs)
            obs_dict = dict_apply(np_obs_dict,
                                    lambda x: torch.from_numpy(np.ascontiguousarray(x)).to(
                                        device=self.device))

            done = False
            traj_reward = 0
            is_success = False
            total_steps = 0
           
            grasp_step = 0
            while not self.grasped and grasp_step < 100:
                action = self.get_action_heuristic(obs_dict)
                action = np.concatenate([action, np.array([self.get_grab_effort(obs_dict)])])
                obs, reward, done, info = env.step([action])
                np_obs_dict = dict[bytes, bytes](obs)
                obs_dict = dict_apply(np_obs_dict,
                                    lambda x: torch.from_numpy(np.ascontiguousarray(x)).to(
                                        device=self.device))
                grasp_step += 1
                total_steps += 1
                
            if not self.grasped:
                done = True
                
            while True:
                if done or bool(info['success']):
                    break
                print("**********, obj_to_target: ", info['obj_to_target'])
                np_obs_dict = dict[bytes, bytes](obs)
                obs_dict = dict_apply(np_obs_dict,
                                      lambda x: torch.from_numpy(np.ascontiguousarray(x)).to(
                                          device=self.device))

                with torch.no_grad():
                    obs_dict_input = {}
                    obs_dict_input['point_cloud'] = obs_dict['point_cloud']
                    obs_dict_input['agent_pos'] = obs_dict['agent_pos']
                    obs_dict_input['instruction'] = self.instruction
                    action = policy.predict_action(obs_dict_input)
                    total_steps += action.shape[0]
                    action[..., -1] = self.get_grab_effort(obs_dict)

                obs, reward, done, info = env.step(action)

                traj_reward += reward
                done = np.all(done)
                is_success = is_success or bool(info['success'])
            
            print(f"\n total_steps: {total_steps}")
            print(f"\n Is success: {is_success}")

            all_success_rates.append(is_success)
            all_traj_rewards.append(traj_reward)
            all_total_steps.append(total_steps)
            # Save video for each episode
            if save_video:
                videos = env.env.get_video()
                if videos is not None and len(videos) > 0:
                    # Save to local file system
                    self._save_video(videos, episode_idx, video_dir)
                    # Store last video for wandb
                    last_videos = videos.copy() if hasattr(videos, 'copy') else videos
            
            # Reset environment for next episode
            _ = env.reset()

        max_rewards = collections.defaultdict(list)
        log_data = dict()

        log_data['mean_traj_rewards'] = np.mean(all_traj_rewards)
        log_data['mean_success_rates'] = np.mean(all_success_rates)

        log_data['test_mean_score'] = np.mean(all_success_rates)
        
        cprint(f"test_mean_score: {np.mean(all_success_rates)}", 'green')
        cprint(f"all_total_steps: {np.mean(all_total_steps)}", 'green')

        self.logger_util_test.record(np.mean(all_success_rates))
        self.logger_util_test10.record(np.mean(all_success_rates))
        log_data['SR_test_L3'] = self.logger_util_test.average_of_largest_K()
        log_data['SR_test_L5'] = self.logger_util_test10.average_of_largest_K()
        
        # Create wandb video object from last episode (if available)
        if save_video and last_videos is not None:
            if len(last_videos.shape) == 5:
                last_videos = last_videos[:, 0]  # select first frame
            
            videos_wandb = wandb.Video(last_videos, fps=self.fps, format="mp4")
            log_data[f'sim_video_eval'] = videos_wandb
            
            # Print wandb storage location if available
            if wandb.run:
                wandb_video_dir = os.path.join(wandb.run.dir, "files", "media")
                cprint(f"WandB video will be stored in: {wandb_video_dir}", 'cyan')
            else:
                cprint("WandB not initialized. Video will be uploaded when wandb.log() is called.", 'yellow')

        # Save data to JSON file
        json_data = {
            'task_name': self.task_name,
            'eval_episodes': self.eval_episodes,
            'episode_data': [
                {
                    'episode_idx': idx,
                    'success_rate': bool(success),  # Convert numpy bool to Python bool
                    'traj_reward': float(reward),   # Convert numpy float to Python float
                    'total_steps': int(steps)       # Convert numpy int to Python int
                }
                for idx, (success, reward, steps) in enumerate(zip(all_success_rates, all_traj_rewards, all_total_steps))
            ],
            'summary_statistics': {
                'mean_traj_rewards': float(log_data['mean_traj_rewards']),
                'mean_success_rates': float(log_data['mean_success_rates']),
                'test_mean_score': float(log_data['test_mean_score']),
                'SR_test_L3': float(log_data['SR_test_L3']),
                'SR_test_L5': float(log_data['SR_test_L5']),
                'mean_total_steps': float(np.mean(all_total_steps))
            }
        }
        
        # Save JSON file
        json_filename = f"eval_results_{self.task_name}.json"
        json_path = os.path.join(self.output_dir, json_filename)
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        cprint(f"Evaluation results saved to: {json_path}", 'green')

        return log_data