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
Adapted from https://github.com/NVlabs/RVT/blob/master/rvt/train.py
Therefore, the code is also under the NVIDIA Source Code License

Author: Peiyan Li
Email: peiyan.li@cripac.ia.ac.cn
'''
# 相较于V2来讲，就是将target pose作为了其中的一个local point
import os
import subprocess
import time
import tqdm
import yaml
import argparse
import time
from collections import defaultdict
from contextlib import redirect_stdout
import torch
import torch.distributed as dist
import deepspeed
import swanlab
os.environ["BITSANDBYTES_NOWELCOME"] = "1"
import bridgevla.config as exp_cfg_mod
import bridgevla.models.bridgevla_agent_3 as bridgevla_agent
import bridgevla.mvt.config as mvt_cfg_mod

from bridgevla.mvt.mvt_3 import MVT
from utils.get_dataset import get_dataset
from bridgevla.utils.rvt_utils import (
    get_num_feat,
    RLBENCH_TASKS,
)
from utils.peract_utils_rlbench import (
    CAMERAS,
    SCENE_BOUNDS,
    IMAGE_SIZE,
    DATA_FOLDER,
    TRAIN_REPLAY_STORAGE_DIR,
)

def train(agent, dataset, training_iterations, epoch, rank=0):
    agent.train()
    log = defaultdict(list)

    data_iter = iter(dataset)
    iter_command = range(training_iterations)
    for iteration in tqdm.tqdm(
        iter_command, disable=(rank != 0), position=0, leave=True
    ):

        raw_batch = next(data_iter)
        batch = {
            k: v.to(agent._device)
            for k, v in raw_batch.items()
            if type(v) == torch.Tensor
        }
        batch["tasks"] = raw_batch["tasks"]
        batch["lang_goal"] = raw_batch["lang_goal"]
        
        # DeepSpeed backward
        update_args = {
            "replay_sample": batch,
            "backprop": True,
            "reset_log": (iteration == 0),
        }
        out = agent.update(**update_args)
        
        if rank == 0:
            step = epoch * training_iterations + iteration
            swanlab.log(out, step=step)
    return log

def save_agent(agent, path, epoch):
    # For DeepSpeed models, use the specific save method
    if hasattr(agent._network, 'module'):
        # DeepSpeed engine - add proper synchronization
        try:
            # Ensure all processes are ready for checkpoint saving
            if dist.is_initialized():
                dist.barrier()
            
            # Save checkpoint with proper error handling
            agent._network.save_checkpoint(os.path.dirname(path), tag=f"epoch_{epoch}")
            
            # Wait for all processes to complete saving
            if dist.is_initialized():
                dist.barrier()
                
            print(f"Successfully saved checkpoint for epoch {epoch}")
            
        except Exception as e:
            print(f"Error saving checkpoint for epoch {epoch}: {e}")
            if dist.is_initialized():
                dist.barrier()  # Ensure all processes continue even if one fails
            raise
    else:
        # Fallback to original save method
        model = agent._network
        model_state = model.state_dict()
        torch.save(
            {
                "epoch": epoch,
                "model_state": model_state,
            },
            path,
        )

def get_tasks(exp_cfg):
    parsed_tasks = exp_cfg.tasks.split(",")
    if parsed_tasks[0] == "all":
        tasks = RLBENCH_TASKS
    else:
        tasks = parsed_tasks
    return tasks

def get_time():
    import datetime
    now = datetime.datetime.now()
    month = now.month
    day = now.day
    hour = now.hour
    minute = now.minute
    folder_name = f"{month:02d}_{day:02d}_{hour:02d}_{minute:02d}"
    return folder_name

def get_logdir(cmd_args, exp_cfg, dist):
    log_dir = os.path.join(cmd_args.log_dir, "train", exp_cfg.exp_id, cmd_args.exp_note)
    if cmd_args.debug == True:
        log_dir = os.path.join(log_dir, "debug")

    if dist.get_rank() == 0:
        os.makedirs(log_dir, exist_ok=True)
    trial_time = get_time()
    log_dir = os.path.join(log_dir, f"{trial_time}")
    if dist.get_rank() == 0:
        os.makedirs(log_dir, exist_ok=True)
    return log_dir

def dump_log(exp_cfg, mvt_cfg, cmd_args, log_dir):
    with open(f"{log_dir}/exp_cfg.yaml", "w") as yaml_file:
        with redirect_stdout(yaml_file):
            print(exp_cfg.dump())

    with open(f"{log_dir}/mvt_cfg.yaml", "w") as yaml_file:
        with redirect_stdout(yaml_file):
            print(mvt_cfg.dump())

    args = cmd_args.__dict__
    with open(f"{log_dir}/args.yaml", "w") as yaml_file:
        yaml.dump(args, yaml_file)

def setup_distributed(backend="nccl", port=None):
    """Initialize distributed training environment.
    support both slurm and torch.distributed.launch
    see torch.distributed.init_process_group() for more details
    """
    num_gpus = torch.cuda.device_count()

    if "SLURM_JOB_ID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        world_size = int(os.environ["SLURM_NTASKS"])
        node_list = os.environ["SLURM_NODELIST"]
        addr = subprocess.getoutput(f"scontrol show hostname {node_list} | head -n1")
        if port is not None:
            os.environ["MASTER_PORT"] = str(port)
        elif "MASTER_PORT" not in os.environ:
            os.environ["MASTER_PORT"] = str(29567 + num_gpus)
        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = addr
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["LOCAL_RANK"] = str(rank % num_gpus)
        os.environ["RANK"] = str(rank)
    else:
        if os.getenv('DEBUG', 'false').lower() == 'true':
            print("Can not find RANK and WORLD_SIZE, Debug Mode")
            os.environ["RANK"] = "0"
            os.environ["WORLD_SIZE"] = "1"
            os.environ["MASTER_ADDR"] = "127.0.0.1"
            os.environ["MASTER_PORT"] = "9001"
            os.environ["LOCAL_RANK"] = "0"
            rank = int(os.environ["RANK"])
            world_size = int(os.environ["WORLD_SIZE"])
        else:
            rank = int(os.environ["RANK"])
            world_size = int(os.environ["WORLD_SIZE"])
    
    # Don't initialize process group here, DeepSpeed will handle it
    return rank, world_size

class DeepSpeedAgent:
    """Wrapper for the agent to work with DeepSpeed"""
    def __init__(self, agent, deepspeed_engine, device):
        self.agent = agent
        self.deepspeed_engine = deepspeed_engine
        self._device = device
        
    def train(self):
        self.deepspeed_engine.train()
        
    def update(self, **kwargs):
        # Modify the agent's update method to work with DeepSpeed
        replay_sample = kwargs['replay_sample']
        backprop = kwargs.get('backprop', True)
        reset_log = kwargs.get('reset_log', False)
        
        # Get the original update method but modify the backward pass
        original_network = self.agent._network
        original_optimizer = self.agent._optimizer
        
        # Temporarily replace network and optimizer
        self.agent._network = self.deepspeed_engine
        self.agent._optimizer = None  # DeepSpeed handles optimization
        
        try:
            # Call the original update but we'll handle backward differently
            if hasattr(self.agent, '_net_mod'):
                original_net_mod = self.agent._net_mod
                self.agent._net_mod = self.deepspeed_engine.module
            
            # Run forward pass and loss computation
            return_out = self._update_with_deepspeed(replay_sample, backprop, reset_log)
            
            return return_out
            
        finally:
            # Restore original references
            self.agent._network = original_network
            self.agent._optimizer = original_optimizer
            if hasattr(self.agent, '_net_mod'):
                self.agent._net_mod = original_net_mod
    
    def _update_with_deepspeed(self, replay_sample, backprop, reset_log):
        # This is a modified version of the original update method
        assert replay_sample["rot_grip_action_indicies"].shape[1:] == (1, 4)
        assert replay_sample["ignore_collisions"].shape[1:] == (1, 1)
        assert replay_sample["gripper_pose"].shape[1:] == (1, 7)

        # Copy the preprocessing from original update method
        action_rot_grip = replay_sample["rot_grip_action_indicies"][:, -1].int()
        action_ignore_collisions = replay_sample["ignore_collisions"][:, -1].int()
        action_gripper_pose = replay_sample["gripper_pose"][:, -1]
        action_trans_con = action_gripper_pose[:, 0:3]
        action_rot = action_gripper_pose[:, 3:7]
        action_grip = action_rot_grip[:, -1]
        tasks = replay_sample["tasks"]
        return_out = {}

        # Import required modules
        import utils.peract_utils_rlbench as rlbench_utils
        import bridgevla.utils.rvt_utils as rvt_utils
        import bridgevla.mvt.utils as mvt_utils
        import bridgevla.mvt.aug_utils as aug_utils
        from bridgevla.models.bridgevla_agent_3 import apply_se3_aug_con, manage_loss_log

        obs, pcd = rlbench_utils._preprocess_inputs(replay_sample, self.agent.cameras)
        
        with torch.no_grad():
            pc, img_feat = rvt_utils.get_pc_img_feat(obs, pcd)

            if self.agent._transform_augmentation and backprop:
                action_trans_con, action_rot, pc = apply_se3_aug_con(
                    pcd=pc,
                    action_gripper_pose=action_gripper_pose,
                    bounds=torch.tensor(self.agent.scene_bounds),
                    trans_aug_range=self.agent._transform_augmentation_xyz.clone().detach(),
                    rot_aug_range=torch.tensor(self.agent._transform_augmentation_rpy),
                )
                action_trans_con = torch.tensor(action_trans_con).to(pc.device)
                action_rot = torch.tensor(action_rot).to(pc.device)

            action_rot = action_rot.cpu().numpy()
            for i, _action_rot in enumerate(action_rot):
                _action_rot = aug_utils.normalize_quaternion(_action_rot)  
                if _action_rot[-1] < 0:
                    _action_rot = -_action_rot
                action_rot[i] = _action_rot
            action_rot = torch.tensor(action_rot).to(pc.device)

            pc, img_feat = rvt_utils.move_pc_in_bound(
                pc, img_feat, self.agent.scene_bounds, no_op=not self.agent.move_pc_in_bound
            )
            # express local points in base frame
            new_action_gripper_pose=torch.cat([action_trans_con, action_rot], dim=-1)
            action_trans_con=rvt_utils.pose_apply_torch(self.agent.points_local, new_action_gripper_pose) #(B,N,3) # 其中一个肯定是原来的gripper pose
            # action_trans_con_with_gt=torch.cat([action_trans_con, new_action_gripper_pose[:,:3].unsqueeze(1)], dim=1) # (B,N+1,3) # 会不会出现内存依赖的情况。
            # wpt = [x[:,:3] for x in action_trans_con_with_gt]  # 已经将多个点引入进来了，后面可能都得跟着修改
            wpt = [x[:,:3] for x in action_trans_con] 

            wpt_local = []
            rev_trans = []
            for _pc, _wpt in zip(pc, wpt):
                a, b = mvt_utils.place_pc_in_cube(
                    _pc,
                    _wpt,
                    with_mean_or_bounds=self.agent._place_with_mean,
                    scene_bounds=None if self.agent._place_with_mean else self.agent.scene_bounds,
                )
                # wpt_local.append(a.unsqueeze(0))
                wpt_local.append(a)
                rev_trans.append(b)

            # wpt_local_with_gt = torch.stack(wpt_local, dim=0) # B,N+1,3
            # wpt_local=wpt_local_with_gt[:,:-1,:] # B,N,3
            wpt_local = torch.stack(wpt_local, dim=0) # B,N,3 # 转换到以工作空间中心为原点的坐标系，且坐标统一*2

            pc = [
                mvt_utils.place_pc_in_cube(
                    _pc,
                    with_mean_or_bounds=self.agent._place_with_mean,
                    scene_bounds=None if self.agent._place_with_mean else self.agent.scene_bounds,
                )[0]
                for _pc in pc
            ]
            


            bs = len(pc)
            nc = self.deepspeed_engine.module.num_img
            h = w = self.deepspeed_engine.module.img_size

            if backprop and (self.agent.img_aug != 0):
                img_aug = self.agent.img_aug
            else:
                img_aug = 0

            dyn_cam_info = None

        (
            action_rot_x_one_hot,
            action_rot_y_one_hot,
            action_rot_z_one_hot,
            action_grip_one_hot,
            action_collision_one_hot,
        ) = self.agent._get_one_hot_expert_actions(
            bs, action_rot, action_grip, action_ignore_collisions, device=self._device
        )

        # if self.agent.rot_ver == 1:
        #     rot_x_y = torch.cat(
        #         [
        #             action_rot_x_one_hot.argmax(dim=-1, keepdim=True),
        #             action_rot_y_one_hot.argmax(dim=-1, keepdim=True),
        #         ],
        #         dim=-1,
        #     )
        #     if self.agent.rot_x_y_aug != 0:
        #         rot_x_y += torch.randint(
        #             -self.agent.rot_x_y_aug, self.agent.rot_x_y_aug, size=rot_x_y.shape
        #         ).to(rot_x_y.device)
        #         rot_x_y %= self.agent._num_rotation_classes
        
        # Forward pass through DeepSpeed engine
        out = self.deepspeed_engine(
            pc=pc,
            img_feat=img_feat,
            lang_emb=None,
            img_aug=img_aug,
            wpt_local=wpt_local if self.deepspeed_engine.training else None,
            # rot_x_y=rot_x_y if self.agent.rot_ver == 1 else None,
            language_goal=replay_sample["lang_goal"],
            ee_points_local=self.agent.points_local,
        )
        
        q_trans, rot_q, grip_q, collision_q, y_q, pts = self.agent.get_q(
            out, dims=(bs, nc,len(self.agent.points_local), h, w)
        )

        action_trans = self.agent.get_action_trans(
           wpt_local, pts, out, dyn_cam_info, dims=(bs, nc,len(self.agent.points_local),h, w)
        )

        loss_log = {}
        if backprop:
            # Cross-entropy loss
            trans_loss = self.agent._cross_entropy_loss(q_trans, action_trans).mean() # 仔细检查一下，到底是在哪一个维度上面去求mean。
            # rot_loss_x = rot_loss_y = rot_loss_z = 0.0
            grip_loss = 0.0
            collision_loss = 0.0
            
            if self.agent.add_rgc_loss:
                # rot_loss_x = self.agent._cross_entropy_loss(
                #     rot_q[:, 0 * self.agent._num_rotation_classes : 1 * self.agent._num_rotation_classes],
                #     action_rot_x_one_hot.argmax(-1),
                # ).mean()

                # rot_loss_y = self.agent._cross_entropy_loss(
                #     rot_q[:, 1 * self.agent._num_rotation_classes : 2 * self.agent._num_rotation_classes],
                #     action_rot_y_one_hot.argmax(-1),
                # ).mean()

                # rot_loss_z = self.agent._cross_entropy_loss(
                #     rot_q[:, 2 * self.agent._num_rotation_classes : 3 * self.agent._num_rotation_classes],
                #     action_rot_z_one_hot.argmax(-1),
                # ).mean()
                
                grip_loss = self.agent._cross_entropy_loss(
                    grip_q, action_grip_one_hot.argmax(-1),
                ).mean()
                
                collision_loss = self.agent._cross_entropy_loss(
                    collision_q, action_collision_one_hot.argmax(-1)
                ).mean()

            total_loss = (
                trans_loss 
                # + rot_loss_x 
                # + rot_loss_y 
                # + rot_loss_z 
                + grip_loss 
                + collision_loss
            )

            # DeepSpeed backward pass
            self.deepspeed_engine.backward(total_loss)
            self.deepspeed_engine.step()

            loss_log = {
                "total_loss": total_loss.item(),
                "trans_loss": trans_loss.item(),
                # "rot_loss_x": rot_loss_x.item(),
                # "rot_loss_y": rot_loss_y.item(),
                # "rot_loss_z": rot_loss_z.item(),
                "grip_loss": grip_loss.item(),
                "collision_loss": collision_loss.item(),
                "lr": self.deepspeed_engine.get_lr()[0],
            }
            manage_loss_log(self.agent, loss_log, reset_log=reset_log)
            return_out.update(loss_log)

        return return_out
        
    def build(self, training=True, device=None):
        return self.agent.build(training=training, device=device)
        
    @property
    def _network(self):
        return self.deepspeed_engine



def experiment(cmd_args):
    rank, world_size = setup_distributed()
    local_rank = int(os.environ["LOCAL_RANK"])
    device_id = f"cuda:{local_rank}"
    
    # Add error handling for GPU setup
    try:
        torch.cuda.set_device(device_id)
        print(f"Rank {rank}: Successfully set device to {device_id}")
    except Exception as e:
        print(f"Rank {rank}: Error setting device {device_id}: {e}")
        raise
    
    exp_cfg = exp_cfg_mod.get_cfg_defaults()

    if cmd_args.exp_cfg_path != "":
        exp_cfg.merge_from_file(cmd_args.exp_cfg_path)
    if cmd_args.exp_cfg_opts != "":
        exp_cfg.merge_from_list(cmd_args.exp_cfg_opts.split(" "))

    old_exp_cfg_peract_lr = exp_cfg.peract.lr
    old_exp_cfg_exp_id = exp_cfg.exp_id
    
    if cmd_args.exp_cfg_opts != "":
        exp_cfg.exp_id += f"_{cmd_args.exp_cfg_opts}"
    if cmd_args.mvt_cfg_opts != "":
        exp_cfg.exp_id += f"_{cmd_args.mvt_cfg_opts}"
    exp_cfg.freeze()

    BATCH_SIZE_TRAIN = exp_cfg.bs
    if local_rank == 0:
        print(f"dict(exp_cfg)={dict(exp_cfg)}")
        print(f"BATCH_SIZE_TRAIN={BATCH_SIZE_TRAIN}")

    NUM_TRAIN = 100
    TRAINING_ITERATIONS = int(exp_cfg.train_iter // (exp_cfg.bs * world_size))

    if exp_cfg.epochs != cmd_args.epochs:
        print(f"cmd args epochs != exp cfg epochs You are using {cmd_args.epochs}")
    EPOCHS = cmd_args.epochs

    data_folder = DATA_FOLDER        
    log_dir = get_logdir(cmd_args, exp_cfg, type('obj', (object,), {'get_rank': lambda: rank}))
    tasks = get_tasks(exp_cfg)
    print("Training on {} tasks: {}".format(len(tasks), tasks))
    
    t_start = time.time()
    get_dataset_func = lambda: get_dataset(
        tasks,
        BATCH_SIZE_TRAIN,
        None,
        TRAIN_REPLAY_STORAGE_DIR,
        None,
        data_folder,
        NUM_TRAIN,
        None,
        cmd_args.refresh_replay,
        device_id,
        num_workers=exp_cfg.num_workers,
        only_train=True,
        sample_distribution_mode=exp_cfg.sample_distribution_mode,
    )
    train_dataset, _ = get_dataset_func()
    t_end = time.time()
    if local_rank == 0:
        print("Created Dataset. Time Cost: {} minutes".format((t_end - t_start) / 60.0))

    mvt_cfg = mvt_cfg_mod.get_cfg_defaults()
    if cmd_args.mvt_cfg_path != "":
        mvt_cfg.merge_from_file(cmd_args.mvt_cfg_path)
    if cmd_args.mvt_cfg_opts != "":
        mvt_cfg.merge_from_list(cmd_args.mvt_cfg_opts.split(" "))

    mvt_cfg.feat_dim = get_num_feat(exp_cfg.peract)
    mvt_cfg.freeze()

    assert mvt_cfg.num_rot == exp_cfg.peract.num_rotation_classes, print(
        mvt_cfg.num_rot, exp_cfg.peract.num_rotation_classes
    )

    num_local_point=50 # hardcode
    backbone = MVT(
        renderer_device=device_id,
        load_pretrain=cmd_args.load_pretrain,
        pretrain_path=cmd_args.pretrain_path,
        num_local_point=num_local_point, # hardcode
        **mvt_cfg,
    )

    agent = bridgevla_agent.RVTAgent(
        network=backbone,
        image_resolution=[IMAGE_SIZE, IMAGE_SIZE],
        stage_two=mvt_cfg.stage_two,
        rot_ver=mvt_cfg.rot_ver,
        scene_bounds=SCENE_BOUNDS,
        cameras=CAMERAS,
        log_dir=f"{log_dir}/test_run/",
        **exp_cfg.peract,
        **exp_cfg.rvt,
    )
    assert num_local_point == len(agent.points_local), print(num_local_point, len(agent.points_local))
    

    freeze_names = ["lm_head", "embed_tokens"]
    if cmd_args.freeze_vision_tower:
        freeze_names.append("vision_tower")
        print("Freeze vision tower")

    for name, module in agent._network.named_modules():
        for freeze_name in freeze_names:
            if freeze_name in name:
                for param in module.parameters():
                    param.requires_grad = False
                break
    
    total_params = sum(p.numel() for p in agent._network.parameters() if p.requires_grad)
    total_params_billion = total_params / 1e9  
    print(f'Total trainable parameters: {total_params_billion:.2f} billion')

    # Initialize DeepSpeed
    model_parameters = filter(lambda p: p.requires_grad, agent._network.parameters())
    
    # Load DeepSpeed config
    import json
    with open(cmd_args.deepspeed_config, 'r') as f:
        ds_config = json.load(f)
    
    # Update batch size in config
    ds_config["train_micro_batch_size_per_gpu"] = BATCH_SIZE_TRAIN
    ds_config["train_batch_size"] = BATCH_SIZE_TRAIN * world_size
    ds_config["optimizer"]["params"]["lr"] = exp_cfg.peract.lr
    
    # Initialize DeepSpeed engine with error handling
    try:
        print(f"Rank {rank}: Initializing DeepSpeed engine...")
        engine, optimizer, _, lr_scheduler = deepspeed.initialize(
            model=agent._network,
            model_parameters=model_parameters,
            config=ds_config
        )
        print(f"Rank {rank}: DeepSpeed engine initialized successfully")
    except Exception as e:
        print(f"Rank {rank}: Error initializing DeepSpeed engine: {e}")
        if dist.is_initialized():
            dist.destroy_process_group()
        raise
    
    # Create DeepSpeed agent wrapper
    agent = DeepSpeedAgent(agent, engine, device_id)
    
    agent.build(training=True, device=device_id)
    start_epoch = 0
    end_epoch = EPOCHS

    if rank == 0:
        temp1 = exp_cfg.peract.lr
        temp2 = exp_cfg.exp_id
        exp_cfg.defrost()
        exp_cfg.peract.lr = old_exp_cfg_peract_lr
        exp_cfg.exp_id = old_exp_cfg_exp_id
        dump_log(exp_cfg, mvt_cfg, cmd_args, log_dir)
        exp_cfg.peract.lr = temp1
        exp_cfg.exp_id = temp2
        exp_cfg.freeze()

    # Initialize Logging =>> SwanLab
    if rank == 0:
        swanlab.login(api_key="h1x6LOLp5qGLTfsPuB7Qw")
        if cmd_args.debug:
            # 如果处在debug模式，我希望就不要上传日志记录
            print("Debug mode, don't upload log record")
            swanlab.init(project="BridgeVLA-V2", experiment_name=os.path.dirname(log_dir), mode="disabled")
        else:
            swanlab.init(project="BridgeVLA-V2", experiment_name=os.path.dirname(log_dir))

    print("Start training ...", flush=True)
    i = start_epoch
    while True:
        if i == end_epoch:
            break

        print(f"Rank [{rank}], Epoch [{i}]: Training on train dataset")

        out = train(agent, train_dataset, TRAINING_ITERATIONS, epoch=i, rank=rank)

        # Save checkpoint with proper synchronization for all processes
        if i % 10 == 0 or i == end_epoch - 1:
            if rank == 0:
                print(f"Saving checkpoint for epoch {i}...")
            save_agent(agent, f"{log_dir}/model_{i}.pth", i)
            save_agent(agent, f"{log_dir}/model_last.pth", i)
        i += 1

    if rank == 0:
        print("[Finish]")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.set_defaults(entry=lambda cmd_args: parser.print_help())
    parser.add_argument("--refresh_replay", action="store_true", default=False)
    parser.add_argument("--mvt_cfg_path", type=str, default="../bridgevla/mvt/configs/rvt2.yaml")
    parser.add_argument("--exp_cfg_path", type=str, default="configs/rlbench_config.yaml")
    parser.add_argument("--mvt_cfg_opts", type=str, default="")
    parser.add_argument("--exp_cfg_opts", type=str, default="")
    parser.add_argument("--exp_note", type=str, default="debug")
    parser.add_argument("--log_dir", type=str, default="logs/")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--freeze_vision_tower", action="store_true")
    # parser.add_argument("--freeze_vision_tower", default=True)
    parser.add_argument("--load_pretrain", action="store_true")
    parser.add_argument("--pretrain_path", type=str, default=None)
    parser.add_argument("--deepspeed_config", type=str, default="configs/deepspeed_config.json")
    parser.add_argument("--local_rank", type=int, default=-1, help="local rank passed from distributed launcher")
    
    cmd_args = parser.parse_args()
    experiment(cmd_args)