import sys

import bridgevla_agent_dm as bridgevla_agent
from utils.metaworld_runner import MetaworldRunnerGrasping
# print(MetaworldRunner.__mro__)
from termcolor import cprint
import os
from utils.setup_paths import setup_project_paths
setup_project_paths()



def eval(args):
    args.output_dir = os.path.join(args.output_dir, args.task)
    env_runner = MetaworldRunnerGrasping(
        output_dir=args.output_dir,
        task_name=args.task,
        device=args.device
    )
    policy = bridgevla_agent.RVTAgent(args)
    runner_log = env_runner.run(policy, save_video=True)

    cprint(f"---------------- Eval Results --------------", 'magenta')
    for key, value in runner_log.items():
          if isinstance(value, float):
               cprint(f"{key}: {value:.4f}", 'magenta')

if __name__ == "__main__":

#     import debugpy
#     debugpy.listen(("0.0.0.0", 5680)) 
#     print("Waiting for debugger attach")
#     debugpy.wait_for_client()

    """主函数"""
    import argparse
    import os

    parser = argparse.ArgumentParser(description='Multi-View Rotation/Gripper Inference Script')

    # 模型配置
    parser.add_argument("--lora_checkpoint", type=str, required=True, help="LoRA checkpoint path for diffusion model")
    parser.add_argument("--rot_grip_checkpoint", type=str, required=True, help="Rotation/gripper predictor checkpoint path")
    parser.add_argument("--model_base_path", type=str, default="/data/lpy/huggingface/Wan2.2-TI2V-5B-fused")
    parser.add_argument("--wan_type", type=str, default="5B_TI2V_RGB_HEATMAP_MV_ROT_GRIP")
    parser.add_argument("--use_dual_head", action='store_true', help='Use dual head mode (must match training configuration)')
    parser.add_argument("--device", type=str, default="cuda")

    # 旋转和夹爪预测器配置
    parser.add_argument("--rotation_resolution", type=float, default=5.0, help="Rotation resolution in degrees")
    parser.add_argument("--hidden_dim", type=int, default=512, help="Hidden dimension")
    parser.add_argument("--num_rotation_bins", type=int, default=72, help="Number of rotation bins (360 / rotation_resolution)")

    # 数据集配置
    parser.add_argument("--scene_bounds", type=str, default="0,-0.45,-0.05,0.8,0.55,0.6",
                       help='Scene bounds as comma-separated values: x_min,y_min,z_min,x_max,y_max,z_max')
    parser.add_argument("--sequence_length", type=int, default=4, help="Sequence length")
    parser.add_argument("--use_merged_pointcloud", action='store_true',
                       help='Use merged pointcloud from 3 cameras (default: False, only use camera 1)')

    # 图像尺寸配置
    parser.add_argument("--img_size", type=str, default="256,256",
                       help='Image size as comma-separated values: height,width (default: 256,256)')
    
     # 输出配置
    parser.add_argument("--output_dir", type=str, default="./output")
    
    # 评估任务名称
    parser.add_argument(
         "--task", type=str, default="door-open"
    )
    
    parser.add_argument(
         "--constant_gripper_num", type=float, default=None,
         help="Constant gripper value to use (if None, use model prediction). Default: None"
    )

    args = parser.parse_args()
    args.scene_bounds = [float(x.strip()) for x in args.scene_bounds.split(',')]
    args.img_size = [int(x.strip()) for x in args.img_size.split(',')]

    eval(args)