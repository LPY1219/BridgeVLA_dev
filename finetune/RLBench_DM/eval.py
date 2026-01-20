'''
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
Adapted from https://github.com/NVlabs/RVT/blob/master/rvt/eval.py
Therefore, the code is also under the NVIDIA Source Code License

Author: Peiyan Li
Email: peiyan.li@cripac.ia.ac.cn
'''
import os
import yaml
import csv
import torch
import cv2
import shutil
import numpy as np
from multiprocessing import Value
from copy import deepcopy

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["BITSANDBYTES_NOWELCOME"] = "1"

from rlbench.backend import task as rlbench_task
from rlbench.backend.utils import task_file_to_task_class
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.action_modes.action_mode import MoveArmThenGripper
from yarr.utils.rollout_generator import RolloutGenerator
from yarr.utils.stat_accumulator import SimpleAccumulator
from yarr.agents.agent import VideoSummary

import bridgevla.mvt.config as default_mvt_cfg
import bridgevla.models.bridgevla_agent as bridgevla_agent
import bridgevla.config as default_exp_cfg

from bridgevla.mvt.mvt import MVT
from bridgevla.libs.peract.helpers import utils
from utils.custom_rlbench_env import (
    CustomMultiTaskRLBenchEnv2 as CustomMultiTaskRLBenchEnv,
)
from utils.peract_utils_rlbench import (
    CAMERAS,
)
from utils.rlbench_planning import (
    EndEffectorPoseViaPlanning2 as EndEffectorPoseViaPlanning,
)
from bridgevla.utils.rvt_utils import (
    TensorboardManager,
    get_eval_parser,
    RLBENCH_TASKS,
)
from bridgevla.utils.rvt_utils import load_agent as load_agent_state
import os
import sys

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))

ENV_IMAGE_SIZE = 256

def load_dm_agent(
    lora_checkpoint,
    rot_grip_checkpoint,
    model_base_path,
    wan_type="5B_TI2V_RGB_HEATMAP_MV_ROT_GRIP",
    use_dual_head=False,
    device=0,
    scene_bounds=None,
    img_size=None,
    sequence_length=4,
    use_merged_pointcloud=True,
    rotation_resolution=5.0,
    hidden_dim=512,
    num_rotation_bins=72,
    constant_gripper_num=None):
    """
    Load DM agent for RLBench evaluation

    Args:
        lora_checkpoint: Path to LoRA checkpoint
        rot_grip_checkpoint: Path to rotation/gripper predictor checkpoint
        model_base_path: Path to base model directory
        wan_type: Model type
        use_dual_head: Whether to use dual head mode
        device: CUDA device id
        scene_bounds: Scene bounds for point cloud processing
        img_size: Image size [H, W]
        sequence_length: Prediction sequence length
        use_merged_pointcloud: Whether to merge point clouds from multiple cameras
        rotation_resolution: Rotation resolution in degrees
        hidden_dim: Hidden dimension for rotation/gripper predictor
        num_rotation_bins: Number of rotation bins
        constant_gripper_num: Constant gripper value (if None, use model prediction)

    Returns:
        agent: DM agent instance
    """
    # Import agent class
    import bridgevla_agent_dm as bridgevla_agent_dm_module

    # Default values
    if scene_bounds is None:
        scene_bounds = [-0.3, -0.5, 0.6, 0.7, 0.5, 1.6]  # RLBench default
    if img_size is None:
        img_size = [256, 256]

    # Create args object
    class Args:
        pass

    args = Args()
    args.lora_checkpoint = lora_checkpoint
    args.rot_grip_checkpoint = rot_grip_checkpoint
    args.model_base_path = model_base_path
    args.wan_type = wan_type
    args.use_dual_head = use_dual_head
    args.device = f"cuda:{device}"
    args.scene_bounds = scene_bounds
    args.img_size = img_size
    args.sequence_length = sequence_length
    args.use_merged_pointcloud = use_merged_pointcloud
    args.rotation_resolution = rotation_resolution
    args.hidden_dim = hidden_dim
    args.num_rotation_bins = num_rotation_bins
    args.constant_gripper_num = constant_gripper_num

    # Create agent
    agent = bridgevla_agent_dm_module.RVTAgent(args)

    print("DM Agent loaded successfully")
    return agent

@torch.no_grad()
def eval_dm(
    agent,
    tasks,
    eval_datafolder,
    start_episode=0,
    eval_episodes=25,
    episode_length=25,
    sequence_length=24,
    device=0,
    headless=True,
    log_dir=None,
    verbose=True,
    save_video=False,
    model_name="debug"):
    """
    Evaluate DM agent on RLBench tasks using rollout generator pattern.

    This function implements action chunking by:
    1. Setting env episode_length to episode_length * sequence_length
    2. Predicting a sequence of actions from the agent
    3. Stepping through each action in the sequence one by one

    Args:
        agent: DM agent instance
        tasks: List of task names to evaluate
        eval_datafolder: Dataset root directory
        start_episode: Starting episode index
        eval_episodes: Number of episodes per task
        episode_length: Maximum prediction steps (number of times we predict actions)
        sequence_length: Number of actions in each predicted sequence
        device: CUDA device id
        headless: Whether to run headless
        log_dir: Log directory
        verbose: Whether to print verbose output
        save_video: Whether to save videos
        model_name: Model name for logging

    Returns:
        scores: List of success rates for each task
    """
    camera_resolution = [ENV_IMAGE_SIZE, ENV_IMAGE_SIZE]
    obs_config = utils.create_obs_config(CAMERAS, camera_resolution, method_name="")

    gripper_mode = Discrete()
    arm_action_mode = EndEffectorPoseViaPlanning()
    action_mode = MoveArmThenGripper(arm_action_mode, gripper_mode)

    task_files = [
        t.replace(".py", "")
        for t in os.listdir(rlbench_task.TASKS_PATH)
        if t != "__init__.py" and t.endswith(".py")
    ]

    task_classes = []
    if tasks[0] == "all":
        tasks = RLBENCH_TASKS
        if verbose:
            print(f"Evaluate on {len(tasks)} tasks: ", tasks)

    for task in tasks:
        if task not in task_files:
            raise ValueError("Task %s not recognised!." % task)
        task_classes.append(task_file_to_task_class(task))
        
    excution_action_step = 20

    # Create environment with extended episode length to accommodate action chunking
    # Each prediction step will execute sequence_length actions
    env_episode_length = episode_length * excution_action_step
    print("Total Env Step:", env_episode_length)

    eval_env = CustomMultiTaskRLBenchEnv(
        task_classes=task_classes,
        observation_config=obs_config,
        action_mode=action_mode,
        dataset_root=eval_datafolder,
        episode_length=env_episode_length,  # Extended length for action chunks
        headless=headless,
        swap_task_every=eval_episodes,
        include_lang_goal_in_obs=True,
        time_in_state=True,
        record_every_n=1 if save_video else -1,
    )

    eval_env.eval = True
    eval_env.launch()

    device_str = f"cuda:{device}"

    # Create logging directories
    if log_dir is not None:
        os.makedirs(log_dir, exist_ok=True)
        csv_file = "eval_results.csv"
        csv_path = os.path.join(log_dir, csv_file)
        if not os.path.exists(csv_path):
            with open(csv_path, "w") as csv_fp:
                fieldnames = ["task", "success rate", "length", "total_transitions"]
                csv_writer = csv.DictWriter(csv_fp, fieldnames=fieldnames)
                csv_writer.writeheader()

    # Create stat accumulator
    stats_accumulator = SimpleAccumulator(eval_video_fps=30)

    scores = []
    num_tasks = len(tasks)

    for task_id in range(num_tasks):
        task_name = tasks[task_id]
        task_rewards = []
        language_goals = []

        if verbose:
            print(f"\n{'='*60}")
            print(f"Evaluating task: {task_name}")
            print(f"{'='*60}")

        for ep in range(start_episode, start_episode + eval_episodes):
            episode_rollout = []

            # Reset environment
            obs_dict = eval_env.reset_to_demo(ep)
            lang_goal = eval_env._lang_goal

            if verbose:
                print(f"\nEpisode {ep} | Lang: {lang_goal}")

            # Action chunking loop: predict once, execute sequence_length times
            for pred_step in range(episode_length):
                print(f"  Prediction step {pred_step} / {episode_length}")
                # Create observation wrapper for agent
                class ObsWrapper:
                    def __init__(self, obs_dict):
                        # Extract point clouds and rgb for all cameras
                        for cam in CAMERAS:
                            pc_key = f'{cam}_point_cloud'
                            rgb_key = f'{cam}_rgb'
                            if pc_key in obs_dict:
                                pc_data = obs_dict[pc_key]
                                if pc_data.ndim == 3 and pc_data.shape[0] == 3:
                                    pc_data = np.transpose(pc_data, (1, 2, 0))
                                setattr(self, pc_key, pc_data)
                            if rgb_key in obs_dict:
                                rgb_data = obs_dict[rgb_key]
                                if rgb_data.ndim == 3 and rgb_data.shape[0] == 3:
                                    rgb_data = np.transpose(rgb_data, (1, 2, 0))
                                setattr(self, rgb_key, rgb_data)

                        # Extract gripper pose and state
                        self.gripper_pose = obs_dict.get('gripper_pose')
                        self.gripper_open = obs_dict.get('gripper_open')

                obs_wrapper = ObsWrapper(obs_dict)

                # Predict action sequence from current observation
                try:
                    action_sequence = agent.predict_action(obs_wrapper, lang_goal)
                except Exception as e:
                    if verbose:
                        print(f"  Prediction error at step {pred_step}: {e}")
                    import traceback
                    traceback.print_exc()
                    break

                # Execute each action in the sequence
                terminal = False
                action_sequence = action_sequence[:excution_action_step]
                for action_idx, act_result in enumerate(action_sequence):
                    # Step environment with single action
                    transition = eval_env.step(act_result)
                    episode_rollout.append(transition)

                    # Update observation for next iteration
                    obs_dict = transition.observation
                    terminal = transition.terminal

                    # Check for success or terminal
                    if terminal:
                        break

                # Break outer loop if episode terminated
                if terminal:
                    break

            # Accumulate statistics for this episode
            for transition in episode_rollout:
                stats_accumulator.step(transition, True)
                current_task_id = transition.info.get("active_task_id", task_id)
                assert current_task_id == task_id

            task_reward = episode_rollout[-1].reward if episode_rollout else 0
            task_rewards.append(task_reward)
            language_goals.append(lang_goal)

            if verbose:
                print(f"Episode {ep} | Score: {task_reward} | Length: {len(episode_rollout)}")

            # Save video immediately after each episode if enabled
            if save_video and log_dir is not None and len(episode_rollout) > 0:
                video_dir = os.path.join(log_dir, "videos")
                os.makedirs(video_dir, exist_ok=True)

                # Collect all video summaries from all transitions in the episode
                video_found = False
                for idx, transition in enumerate(episode_rollout):
                    if transition.summaries:
                        for summary in transition.summaries:
                            if isinstance(summary, VideoSummary):
                                video_found = True
                                lang_goal_clean = lang_goal.replace(" ", "_")
                                video = deepcopy(summary.value)
                                video = np.transpose(video, (0, 2, 3, 1))  # (T, H, W, C)
                                video = video[:, :, :, ::-1]  # RGB to BGR
                                
                                # Remove last 10 frames (success/failure indicator frames)
                                # These are solid green/red frames added by _append_final_frame
                                if len(video) > 10:
                                    video = video[:-10]

                                # Determine success/failure
                                is_success = task_reward > 99

                                if is_success:
                                    video_filename = f"{task_name}_ep{ep}_{lang_goal_clean}_success.mp4"
                                else:
                                    video_filename = f"{task_name}_ep{ep}_{lang_goal_clean}_fail.mp4"

                                video_path = os.path.join(video_dir, video_filename)

                                # Save video using ffmpeg
                                if len(video) > 0:
                                    # Use unique tmp folder for each episode to avoid conflicts in multi-GPU setup
                                    import time
                                    import random
                                    unique_id = f"{task_name}_ep{ep}_{int(time.time())}_{random.randint(1000, 9999)}"
                                    tmp_folder = f"./tmp/{model_name}/{unique_id}"
                                    os.makedirs(tmp_folder, exist_ok=True)

                                    # Save all frames
                                    for idx in range(len(video)):
                                        cv2.imwrite(os.path.join(tmp_folder, f"{idx}.png"), video[idx])

                                    images_path = os.path.join(tmp_folder, r"%d.png")
                                    # Use episode-specific palette file to avoid conflicts
                                    palette_path = os.path.join(tmp_folder, "palette.png")

                                    # Generate palette and video (suppress warnings)
                                    os.system(
                                        f"ffmpeg -i {images_path} -vf palettegen {palette_path} -hide_banner -loglevel error 2>/dev/null"
                                    )
                                    os.system(
                                        f"ffmpeg -framerate 25 -i {images_path} -i {palette_path} -lavfi paletteuse {video_path} -hide_banner -loglevel error 2>/dev/null"
                                    )

                                    # Cleanup entire tmp folder
                                    if os.path.exists(tmp_folder):
                                        shutil.rmtree(tmp_folder)

                                    if verbose:
                                        print(f"  Video saved: {video_path}")

                                # Only save the first video summary found
                                break
                    if video_found:
                        break

                if not video_found and verbose:
                    print(f"  Warning: No video summary found for episode {ep}")

        # Report summaries
        summaries = []
        summaries.extend(stats_accumulator.pop())

        if log_dir is not None:
            # Write to CSV
            with open(csv_path, "a") as csv_fp:
                fieldnames = ["task", "success rate", "length", "total_transitions"]
                csv_writer = csv.DictWriter(csv_fp, fieldnames=fieldnames)
                csv_results = {"task": task_name}
                for s in summaries:
                    if s.name == "eval_envs/return":
                        csv_results["success rate"] = s.value
                    elif s.name == "eval_envs/length":
                        csv_results["length"] = s.value
                    elif s.name == "eval_envs/total_transitions":
                        csv_results["total_transitions"] = s.value
                    if "eval" in s.name:
                        s.name = f"{s.name}/{task_name}"
                csv_writer.writerow(csv_results)
        else:
            for s in summaries:
                if "eval" in s.name:
                    s.name = f"{s.name}/{task_name}"

        if len(summaries) > 0:
            task_score = [
                s.value for s in summaries if f"eval_envs/return/{task_name}" in s.name
            ][0]
        else:
            task_score = 0.0

        print(f"[Evaluation] Finished {task_name} | Final Score: {task_score}\n")
        scores.append(task_score)

    eval_env.shutdown()
    return scores


def _eval_dm(args):
    """
    Main evaluation function for DM agent
    """
    # Load DM agent
    agent = load_dm_agent(
        lora_checkpoint=args.lora_checkpoint,
        rot_grip_checkpoint=args.rot_grip_checkpoint,
        model_base_path=args.model_base_path,
        wan_type=args.wan_type,
        use_dual_head=args.use_dual_head,
        device=args.device,
        scene_bounds=args.scene_bounds,
        img_size=args.img_size,
        sequence_length=args.sequence_length,
        use_merged_pointcloud=args.use_merged_pointcloud,
        rotation_resolution=args.rotation_resolution,
        hidden_dim=args.hidden_dim,
        num_rotation_bins=args.num_rotation_bins,
        constant_gripper_num=args.constant_gripper_num,
    )

    # Run evaluation
    scores = eval_dm(
        agent=agent,
        tasks=args.tasks,
        eval_datafolder=args.eval_datafolder,
        start_episode=args.start_episode,
        eval_episodes=args.eval_episodes,
        episode_length=args.episode_length,
        sequence_length=args.sequence_length,
        device=args.device,
        headless=args.headless,
        log_dir=args.eval_log_dir,
        verbose=True,
        save_video=args.save_video,
        model_name=args.log_name,
    )

    print(f"\n{'='*60}")
    print(f"Overall Results:")
    print(f"{'='*60}")
    for i, task in enumerate(args.tasks):
        print(f"{task}: {scores[i]:.2%}")
    print(f"Average: {sum(scores)/len(scores):.2%}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    # import debugpy
    # debugpy.listen(("0.0.0.0", 5678)) 
    # print("Waiting for debugger attach")
    # debugpy.wait_for_client()


    import argparse
    import os

    parser = argparse.ArgumentParser(description='Multi-View Rotation/Gripper Inference Script for RLBench')

    # 模型配置
    parser.add_argument("--lora_checkpoint", type=str, required=True, help="LoRA checkpoint path for diffusion model")
    parser.add_argument("--rot_grip_checkpoint", type=str, required=True, help="Rotation/gripper predictor checkpoint path")
    parser.add_argument("--model_base_path", type=str, default="/data/lpy/huggingface/Wan2.2-TI2V-5B-fused")
    parser.add_argument("--model_folder", type=str, default="./", help="Model folder path")
    parser.add_argument("--model_name", type=str, default=None, help="Model name")
    parser.add_argument("--wan_type", type=str, default="5B_TI2V_RGB_HEATMAP_MV_ROT_GRIP")
    parser.add_argument("--use_dual_head", action='store_true', help='Use dual head mode (must match training configuration)')
    parser.add_argument("--device", type=int, default=0, help="CUDA device id")

    # 评估配置参数 (from RLBench/eval.py)
    parser.add_argument("--eval_datafolder", type=str, required=True, help="Path to evaluation dataset")
    parser.add_argument("--tasks", nargs='+', type=str, default=["push_buttons"], help="List of tasks to evaluate")
    parser.add_argument("--eval_episodes", type=int, default=25, help="Number of episodes per task")
    parser.add_argument("--episode_length", type=int, default=25, help="Maximum steps per episode")
    parser.add_argument("--start_episode", type=int, default=0, help="Starting episode index")
    parser.add_argument("--headless", action='store_true', help="Run in headless mode")
    parser.add_argument("--log_name", type=str, default=None, help="Log directory name")
    parser.add_argument("--ground_truth", action='store_true', help="Replay ground truth actions")
    parser.add_argument("--save_video", action='store_true', help="Save evaluation videos")
    parser.add_argument("--visualize", action='store_true', help="Enable visualization")
    parser.add_argument("--visualize_root_dir", type=str, default="./visualize", help="Visualization root directory")

    # 旋转和夹爪预测器配置
    parser.add_argument("--rotation_resolution", type=float, default=5.0, help="Rotation resolution in degrees")
    parser.add_argument("--hidden_dim", type=int, default=512, help="Hidden dimension")
    parser.add_argument("--num_rotation_bins", type=int, default=72, help="Number of rotation bins (360 / rotation_resolution)")

    # 数据集配置
    parser.add_argument("--scene_bounds", nargs='+', type=float, default=[-0.3, -0.5, 0.6, 0.7, 0.5, 1.6],
                       help='Scene bounds as 6 values: x_min y_min z_min x_max y_max z_max')
    parser.add_argument("--sequence_length", type=int, default=24, help="Sequence length")
    parser.add_argument("--use_merged_pointcloud", action='store_true',
                       help='Use merged pointcloud from 3 cameras (default: False, only use camera 1)')

    # 图像尺寸配置
    parser.add_argument("--img_size", nargs='+', type=int, default=[256, 256],
                       help='Image size as 2 values: height width (default: 256 256)')

    # 输出配置
    parser.add_argument("--output_dir", type=str, default="./output")

    parser.add_argument("--constant_gripper_num", type=float, default=None,
         help="Constant gripper value to use (if None, use model prediction). Default: None")

    args = parser.parse_args()

    # Validate scene_bounds length
    if len(args.scene_bounds) != 6:
        raise ValueError(f"scene_bounds must have 6 values, got {len(args.scene_bounds)}")

    # Validate img_size length
    if len(args.img_size) != 2:
        raise ValueError(f"img_size must have 2 values, got {len(args.img_size)}")

    if args.log_name is None:
        args.log_name = "none"

    args.eval_log_dir = os.path.join(args.model_folder, "eval", args.log_name)

    os.makedirs(args.eval_log_dir, exist_ok=True)

    # save the arguments for future reference
    with open(os.path.join(args.eval_log_dir, "eval_config.yaml"), "w") as fp:
        yaml.dump(args.__dict__, fp)

    _eval_dm(args)

