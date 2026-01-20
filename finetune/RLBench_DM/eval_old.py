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
    CustomMultiTaskRLBenchEnvActionChunk as CustomMultiTaskRLBenchEnvActionChunk,
)
from utils.peract_utils_rlbench import (
    CAMERAS,
    SCENE_BOUNDS,
    IMAGE_SIZE,
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
    device=0,
    headless=True,
    log_dir=None,
    verbose=True,
    save_video=False,
    model_name="debug"):
    """
    Evaluate DM agent on RLBench tasks

    Args:
        agent: DM agent instance
        tasks: List of task names to evaluate
        eval_datafolder: Dataset root directory
        start_episode: Starting episode index
        eval_episodes: Number of episodes per task
        episode_length: Maximum steps per episode
        device: CUDA device id
        headless: Whether to run headless
        log_dir: Log directory
        verbose: Whether to print verbose output

    Returns:
        scores: List of success rates for each task
    """
    camera_resolution = [IMAGE_SIZE, IMAGE_SIZE]
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

    # Set record_every_n based on save_video flag
    record_every_n = 1 if save_video else -1
    eval_env = CustomMultiTaskRLBenchEnvActionChunk(
        task_classes=task_classes,
        observation_config=obs_config,
        action_mode=action_mode,
        dataset_root=eval_datafolder,
        episode_length=episode_length,
        headless=headless,
        swap_task_every=eval_episodes,
        include_lang_goal_in_obs=True,
        time_in_state=True,
        record_every_n=record_every_n,
    )

    eval_env.eval = True
    eval_env.launch()
    device = f"cuda:{device}"
    # Create logging directories
    if log_dir is not None:
        os.makedirs(log_dir, exist_ok=True)
        csv_file = "eval_results.csv"
        csv_path = os.path.join(log_dir, csv_file)
        if not os.path.exists(csv_path):
            with open(csv_path, "w") as csv_fp:
                fieldnames = ["task", "success_rate", "avg_steps"]
                csv_writer = csv.DictWriter(csv_fp, fieldnames=fieldnames)
                csv_writer.writeheader()

        # Create video directory if saving videos
        if save_video:
            video_dir = os.path.join(log_dir, "videos")
            os.makedirs(video_dir, exist_ok=True)

    # # Create rollout generator for handling episodes
    # rollout_generator = RolloutGenerator()

    # # Create stat accumulator for tracking metrics
    # stat_accumulator = SimpleAccumulator(eval_video_fps=30)

    scores = []
    num_tasks = len(tasks)

    for task_id in range(num_tasks):
        task_name = tasks[task_id]
        task_successes = []
        task_steps = []

        if verbose:
            print(f"\n{'='*60}")
            print(f"Evaluating task: {task_name}")
            print(f"{'='*60}")

        for ep in range(start_episode, start_episode + eval_episodes):
            # Reset environment
            obs_dict = eval_env.reset_to_demo(ep)
            lang_goal = eval_env._lang_goal

            if verbose:
                print(f"\nEpisode {ep}/{start_episode + eval_episodes - 1}")
                print(f"Language goal: {lang_goal}")

            episode_success = False
            step_count = 0
            episode_transitions = []

            for step in range(episode_length):
                step_count += 1

                try:
                    # Create Observation-like object with necessary attributes
                    class ObsWrapper:
                        def __init__(self, obs_dict):
                            # Extract point clouds and rgb for all cameras
                            for cam in CAMERAS:
                                pc_key = f'{cam}_point_cloud'
                                rgb_key = f'{cam}_rgb'
                                if pc_key in obs_dict:
                                    pc_data = obs_dict[pc_key]
                                    if pc_data.ndim==3 and pc_data.shape[0]==3:
                                        pc_data = np.transpose(pc_data, (1, 2, 0))
                                    setattr(self, pc_key, pc_data)
                                if rgb_key in obs_dict:
                                    rgb_data = obs_dict[rgb_key]
                                    # Reshape from (3, H, W) to (H, W, 3)
                                    if rgb_data.ndim == 3 and rgb_data.shape[0] == 3:
                                        rgb_data = np.transpose(rgb_data, (1, 2, 0))
                                    setattr(self, rgb_key, rgb_data)

                            # Extract gripper pose and state
                            self.gripper_pose = obs_dict.get('gripper_pose')
                            self.gripper_open = obs_dict.get('gripper_open')

                    obs_wrapper = ObsWrapper(obs_dict)

                    # Predict action sequence (returns list of ActResult)
                    actions = agent.predict_action(obs_wrapper, lang_goal)
                    actions = actions[:20]  # Take first five action in sequence

                    # Apply action to environment - returns Transition
                    transition = eval_env.step(actions)
                    episode_transitions.append(transition)

                    # Extract obs and reward from transition
                    obs_dict = transition.observation
                    reward = transition.reward
                    terminal = transition.terminal
                    print("Reward:", reward)
                    if reward >= 1:
                        episode_success = True
                        if verbose:
                            print(f"  Step {step}: Success!")
                        break

                    if terminal:
                        if verbose:
                            print(f"  Step {step}: Terminal")
                        break

                except Exception as e:
                    if verbose:
                        print(f"  Error at step {step}: {e}")
                    import traceback
                    traceback.print_exc()
                    break

            task_successes.append(episode_success)
            task_steps.append(step_count)

            if verbose:
                print(f"  Episode {ep} result: {'Success' if episode_success else 'Failure'} (steps: {step_count})")

            # Save video if enabled and summaries are available
            if save_video and log_dir is not None and len(episode_transitions) > 0:
                # Get summaries from last transition
                last_transition = episode_transitions[-1]
                if last_transition.summaries:
                    for summary in last_transition.summaries:
                        if isinstance(summary, VideoSummary):
                            # Save video
                            video_filename = f"{task_name}_ep{ep}_{'success' if episode_success else 'fail'}.mp4"
                            video_path = os.path.join(video_dir, video_filename)

                            # Convert video tensor to format for saving
                            # summary.value shape: (T, C, H, W)
                            video_data = summary.value  # (T, C, H, W)

                            # Save using cv2
                            if video_data.shape[0] > 0:
                                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                                height, width = video_data.shape[2], video_data.shape[3]
                                out = cv2.VideoWriter(video_path, fourcc, 30.0, (width, height))

                                for frame_idx in range(video_data.shape[0]):
                                    # Convert from (C, H, W) to (H, W, C)
                                    frame = video_data[frame_idx].transpose(1, 2, 0)
                                    # Convert RGB to BGR for cv2
                                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                                    out.write(frame_bgr)

                                out.release()
                                if verbose:
                                    print(f"  Saved video to: {video_path}")

        # Calculate task metrics
        success_rate = sum(task_successes) / len(task_successes)
        avg_steps = sum(task_steps) / len(task_steps)

        scores.append(success_rate)

        if verbose:
            print(f"\n{'='*60}")
            print(f"Task {task_name} Results:")
            print(f"  Success Rate: {success_rate:.2%} ({sum(task_successes)}/{len(task_successes)})")
            print(f"  Average Steps: {avg_steps:.1f}")
            print(f"{'='*60}\n")

        # Log to CSV
        if log_dir is not None:
            with open(csv_path, "a") as csv_fp:
                csv_writer = csv.DictWriter(csv_fp, fieldnames=["task", "success_rate", "avg_steps"])
                csv_writer.writerow({
                    "task": task_name,
                    "success_rate": success_rate,
                    "avg_steps": avg_steps
                })

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
    parser.add_argument("--sequence_length", type=int, default=4, help="Sequence length")
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

