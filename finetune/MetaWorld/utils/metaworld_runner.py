import wandb
from typing import Dict, Callable
import numpy as np
import torch
import collections
import tqdm
import os
from utils.env import MetaWorldEnv
from utils.multistep_wrapper import MultiStepWrapper
from utils.video_recording_wrapper import SimpleVideoRecordingWrapper

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
            traj_reward = 0
            is_success = False
            while not done:
                np_obs_dict = dict[bytes, bytes](obs)
                obs_dict = dict_apply(np_obs_dict,
                                      lambda x: torch.from_numpy(x).to(
                                          device=self.device))

                with torch.no_grad():
                    obs_dict_input = {}
                    obs_dict_input['point_cloud'] = obs_dict['point_cloud']
                    obs_dict_input['agent_pos'] = obs_dict['agent_pos']
                    obs_dict_input['instruction'] = self.instruction
                    action = policy.predict_action(obs_dict_input)

                obs, reward, done, info = env.step(action)

                traj_reward += reward
                done = np.all(done)
                is_success = is_success or max(info['success'])

            all_success_rates.append(is_success)
            all_traj_rewards.append(traj_reward)
            
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

        return log_data
