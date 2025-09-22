"""
训练配置文件
包含学习率、批次大小、优化器设置等训练相关配置
"""

import os
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
import torch


@dataclass
class OptimizerConfig:
    """
    优化器配置
    """
    # 优化器类型
    optimizer_type: str = "adamw"  # "adam", "adamw", "sgd"

    # 学习率相关
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8

    # SGD特有参数
    momentum: float = 0.9
    nesterov: bool = True

    # 学习率调度器
    scheduler_type: str = "cosine"  # "cosine", "step", "plateau", "linear", "constant"
    warmup_steps: int = 1000
    warmup_ratio: float = 0.1
    min_lr_ratio: float = 0.01

    # StepLR特有参数
    step_size: int = 30
    gamma: float = 0.1

    # ReduceLROnPlateau特有参数
    patience: int = 10
    factor: float = 0.5
    threshold: float = 1e-4

    def get_optimizer(self, parameters):
        """创建优化器"""
        if self.optimizer_type.lower() == "adam":
            return torch.optim.Adam(
                parameters,
                lr=self.learning_rate,
                betas=(self.beta1, self.beta2),
                eps=self.eps,
                weight_decay=self.weight_decay
            )
        elif self.optimizer_type.lower() == "adamw":
            return torch.optim.AdamW(
                parameters,
                lr=self.learning_rate,
                betas=(self.beta1, self.beta2),
                eps=self.eps,
                weight_decay=self.weight_decay
            )
        elif self.optimizer_type.lower() == "sgd":
            return torch.optim.SGD(
                parameters,
                lr=self.learning_rate,
                momentum=self.momentum,
                weight_decay=self.weight_decay,
                nesterov=self.nesterov
            )
        else:
            raise ValueError(f"Unknown optimizer type: {self.optimizer_type}")

    def get_scheduler(self, optimizer, num_training_steps: int):
        """创建学习率调度器"""
        if self.scheduler_type.lower() == "cosine":
            try:
                from transformers import get_cosine_schedule_with_warmup
                return get_cosine_schedule_with_warmup(
                    optimizer,
                    num_warmup_steps=self.warmup_steps,
                    num_training_steps=num_training_steps
                )
            except ImportError:
                print("Warning: transformers not available, using CosineAnnealingLR")
                return torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=num_training_steps,
                    eta_min=self.learning_rate * self.min_lr_ratio
                )
        elif self.scheduler_type.lower() == "step":
            return torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=self.step_size,
                gamma=self.gamma
            )
        elif self.scheduler_type.lower() == "plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=self.factor,
                patience=self.patience,
                threshold=self.threshold,
                verbose=True
            )
        elif self.scheduler_type.lower() == "linear":
            try:
                from transformers import get_linear_schedule_with_warmup
                return get_linear_schedule_with_warmup(
                    optimizer,
                    num_warmup_steps=self.warmup_steps,
                    num_training_steps=num_training_steps
                )
            except ImportError:
                print("Warning: transformers not available, using LinearLR")
                return torch.optim.lr_scheduler.LinearLR(
                    optimizer,
                    start_factor=0.1,
                    total_iters=self.warmup_steps
                )
        elif self.scheduler_type.lower() == "constant":
            return torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)
        else:
            raise ValueError(f"Unknown scheduler type: {self.scheduler_type}")


@dataclass
class LossConfig:
    """
    损失函数配置
    """
    # 主要损失函数类型
    primary_loss: str = "mse"  # "mse", "l1", "huber", "focal"

    # 损失权重
    mse_weight: float = 1.0
    peak_loss_weight: float = 2.0  # 峰值位置损失权重
    consistency_loss_weight: float = 0.5  # 序列一致性损失权重
    perceptual_loss_weight: float = 0.1  # 感知损失权重

    # Focal Loss参数
    focal_alpha: float = 1.0
    focal_gamma: float = 2.0

    # Huber Loss参数
    huber_delta: float = 1.0

    # 峰值损失参数
    peak_loss_sigma: float = 2.0  # 峰值损失的高斯权重
    peak_loss_threshold: float = 0.1  # 峰值检测阈值

    # 一致性损失参数
    consistency_method: str = "peak_distance"  # "peak_distance", "mse", "correlation"

    def get_loss_function(self):
        """获取损失函数"""
        if self.primary_loss.lower() == "mse":
            return torch.nn.MSELoss()
        elif self.primary_loss.lower() == "l1":
            return torch.nn.L1Loss()
        elif self.primary_loss.lower() == "huber":
            return torch.nn.HuberLoss(delta=self.huber_delta)
        else:
            raise ValueError(f"Unknown loss type: {self.primary_loss}")


@dataclass
class TrainingConfig:
    """
    训练相关配置
    """
    # 训练基本参数
    num_epochs: int = 100
    max_steps: Optional[int] = None  # 如果设置，会覆盖num_epochs
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0  # 梯度裁剪

    # 评估和保存
    eval_steps: int = 500  # 每多少步评估一次
    save_steps: int = 1000  # 每多少步保存一次
    logging_steps: int = 100  # 每多少步记录日志
    save_total_limit: int = 5  # 最多保存几个检查点

    # 早停
    early_stopping: bool = True
    early_stopping_patience: int = 20
    early_stopping_threshold: float = 1e-4

    # 检查点和恢复
    resume_from_checkpoint: Optional[str] = None
    save_checkpoint_on_interrupt: bool = True

    # 验证相关
    validation_ratio: float = 0.1  # 验证集比例
    metric_for_best_model: str = "val_loss"  # "val_loss", "val_accuracy", "val_mse"
    greater_is_better: bool = False  # 对于loss，False；对于accuracy，True

    # 其他训练设置
    dataloader_num_workers: int = 4
    dataloader_pin_memory: bool = False  # 设为False避免CUDA张量pin_memory冲突
    seed: int = 42

    # 混合精度训练
    fp16: bool = False
    bf16: bool = False

    # 分布式训练
    local_rank: int = -1
    ddp_backend: str = "nccl"

    def get_total_steps(self, dataset_size: int, batch_size: int) -> int:
        """计算总训练步数"""
        if self.max_steps is not None:
            return self.max_steps

        steps_per_epoch = dataset_size // (batch_size * self.gradient_accumulation_steps)
        return steps_per_epoch * self.num_epochs


@dataclass
class LoggingConfig:
    """
    日志和监控配置
    """
    # 基本日志设置
    log_level: str = "INFO"  # "DEBUG", "INFO", "WARNING", "ERROR"
    log_file: Optional[str] = None
    console_log: bool = True

    # TensorBoard
    use_tensorboard: bool = True
    tensorboard_log_dir: str = "./logs/tensorboard"

    # Weights & Biases
    use_wandb: bool = False
    wandb_project: str = "heatmap_prediction"
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = None
    wandb_tags: List[str] = field(default_factory=list)

    # 其他监控工具
    use_neptune: bool = False
    use_comet: bool = False

    # 日志内容
    log_model_architecture: bool = True
    log_gradients: bool = False
    log_weights: bool = False
    log_images: bool = True  # 是否记录样本图像
    log_frequency: int = 100  # 记录频率


@dataclass
class ValidationConfig:
    """
    验证配置
    """
    # 验证频率
    validate_every_n_epochs: int = 1
    validate_every_n_steps: Optional[int] = None

    # 验证指标
    compute_metrics: List[str] = field(default_factory=lambda: [
        "mse", "peak_accuracy", "sequence_consistency"
    ])

    # 峰值精度评估
    peak_accuracy_thresholds: List[float] = field(default_factory=lambda: [2.0, 5.0, 10.0])

    # 可视化验证结果
    visualize_predictions: bool = True
    num_visualization_samples: int = 4
    save_visualization: bool = True

    # 验证时的推理设置
    validation_inference_steps: int = 20  # 验证时使用更少的推理步数以加速


@dataclass
class CompleteTrainingConfig:
    """
    完整的训练配置，整合所有子配置
    """
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'optimizer': self.optimizer.__dict__,
            'loss': self.loss.__dict__,
            'training': self.training.__dict__,
            'logging': self.logging.__dict__,
            'validation': self.validation.__dict__
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]):
        """从字典创建配置对象"""
        return cls(
            optimizer=OptimizerConfig(**config_dict.get('optimizer', {})),
            loss=LossConfig(**config_dict.get('loss', {})),
            training=TrainingConfig(**config_dict.get('training', {})),
            logging=LoggingConfig(**config_dict.get('logging', {})),
            validation=ValidationConfig(**config_dict.get('validation', {}))
        )

    def save(self, path: str):
        """保存配置到文件"""
        import json
        config_dict = self.to_dict()
        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=2)
        print(f"Training config saved to: {path}")

    @classmethod
    def load(cls, path: str):
        """从文件加载配置"""
        import json
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)


# 预定义训练配置模板

def get_default_training_config() -> CompleteTrainingConfig:
    """获取默认训练配置"""
    return CompleteTrainingConfig()


def get_debug_training_config() -> CompleteTrainingConfig:
    """获取调试训练配置（快速训练）"""
    config = CompleteTrainingConfig()

    # 快速训练设置
    config.training.num_epochs = 5
    config.training.eval_steps = 50
    config.training.save_steps = 100
    config.training.logging_steps = 10

    # 简化优化器
    config.optimizer.learning_rate = 1e-3
    config.optimizer.warmup_steps = 50

    # 简化验证
    config.validation.num_visualization_samples = 2
    config.validation.validation_inference_steps = 10

    # 简化日志
    config.logging.use_tensorboard = False
    config.logging.use_wandb = False

    return config


def get_production_training_config() -> CompleteTrainingConfig:
    """获取生产环境训练配置（完整训练）"""
    config = CompleteTrainingConfig()

    # 完整训练设置
    config.training.num_epochs = 200
    config.training.eval_steps = 1000
    config.training.save_steps = 2000
    config.training.early_stopping_patience = 30

    # 精细调优的优化器
    config.optimizer.learning_rate = 5e-5
    config.optimizer.scheduler_type = "cosine"
    config.optimizer.warmup_steps = 2000

    # 更复杂的损失函数
    config.loss.peak_loss_weight = 3.0
    config.loss.consistency_loss_weight = 1.0
    config.loss.perceptual_loss_weight = 0.2

    # 完整监控
    config.logging.use_tensorboard = True
    config.logging.use_wandb = True
    config.logging.log_gradients = True
    config.logging.log_images = True

    # 混合精度
    config.training.fp16 = True

    return config


def get_fine_tuning_config() -> CompleteTrainingConfig:
    """获取微调配置（从预训练模型微调）"""
    config = CompleteTrainingConfig()

    # 微调设置
    config.training.num_epochs = 50
    config.training.eval_steps = 200
    config.training.save_steps = 500

    # 较小的学习率
    config.optimizer.learning_rate = 1e-5
    config.optimizer.weight_decay = 1e-5
    config.optimizer.scheduler_type = "linear"
    config.optimizer.warmup_steps = 500

    # 更注重峰值精度
    config.loss.peak_loss_weight = 5.0
    config.loss.mse_weight = 0.5

    return config


def test_training_config():
    """测试训练配置功能"""
    print("Testing training configuration...")

    # 测试默认配置
    config = get_default_training_config()
    print(f"Default config epochs: {config.training.num_epochs}")
    print(f"Default config learning rate: {config.optimizer.learning_rate}")

    # 测试优化器创建
    dummy_params = [torch.randn(10, 10, requires_grad=True)]
    optimizer = config.optimizer.get_optimizer(dummy_params)
    print(f"Optimizer type: {type(optimizer).__name__}")

    # 测试学习率调度器
    scheduler = config.optimizer.get_scheduler(optimizer, num_training_steps=1000)
    print(f"Scheduler type: {type(scheduler).__name__}")

    # 测试损失函数
    loss_fn = config.loss.get_loss_function()
    print(f"Loss function: {type(loss_fn).__name__}")

    # 测试保存和加载
    test_path = "/tmp/test_training_config.json"
    config.save(test_path)
    loaded_config = CompleteTrainingConfig.load(test_path)
    print(f"Loaded config epochs: {loaded_config.training.num_epochs}")

    # 测试预定义配置
    debug_config = get_debug_training_config()
    print(f"Debug config epochs: {debug_config.training.num_epochs}")

    prod_config = get_production_training_config()
    print(f"Production config epochs: {prod_config.training.num_epochs}")

    finetune_config = get_fine_tuning_config()
    print(f"Fine-tuning config learning rate: {finetune_config.optimizer.learning_rate}")

    print("Training config test completed!")
    return True


if __name__ == "__main__":
    test_training_config()