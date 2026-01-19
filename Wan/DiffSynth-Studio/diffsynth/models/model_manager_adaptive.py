"""
Adaptive Model Manager
支持自动维度适配的 ModelManager，用于加载预训练权重时自动处理维度变化
"""

import os
import torch
from .model_manager import ModelManager as BaseModelManager
from .model_manager import load_state_dict, init_weights_on_device
import diffsynth.models.model_manager as model_manager_module


def load_model_from_single_file_adaptive(state_dict, model_names, model_classes, model_resource, torch_dtype, device, use_dual_head=False):
    """
    加载模型的增强版本，支持自动维度适配

    当模型的 in_dim 或 out_dim 与预训练权重不匹配时，
    会自动调用 adapt_pretrained_weights 方法进行适配

    Args:
        use_dual_head: 是否使用双head模式（为RGB和Heatmap分别使用独立的head）
    """
    loaded_model_names, loaded_models = [], []
    for model_name, model_class in zip(model_names, model_classes):
        print(f"    model_name: {model_name} model_class: {model_class.__name__}")
        state_dict_converter = model_class.state_dict_converter()
        if model_resource == "civitai":
            state_dict_results = state_dict_converter.from_civitai(state_dict)
        elif model_resource == "diffusers":
            state_dict_results = state_dict_converter.from_diffusers(state_dict)

        if isinstance(state_dict_results, tuple):
            model_state_dict, extra_kwargs = state_dict_results
            print(f"        This model is initialized with extra kwargs: {extra_kwargs}")
        else:
            model_state_dict, extra_kwargs = state_dict_results, {}

        # 添加 use_dual_head 参数到 extra_kwargs（如果模型支持）
        model_init_params = model_class.__init__.__code__.co_varnames
        print(f"        [DEBUG] Model init parameters for {model_class.__name__}: {model_init_params[:15]}")  # 打印前15个参数
        if 'use_dual_head' in model_init_params:
            extra_kwargs['use_dual_head'] = use_dual_head
            print(f"        ✓ Setting use_dual_head={use_dual_head} for {model_class.__name__}")
        else:
            print(f"        ✗ use_dual_head not in {model_class.__name__} init params")

        torch_dtype = torch.float32 if extra_kwargs.get("upcast_to_float32", False) else torch_dtype

        with init_weights_on_device():
            model = model_class(**extra_kwargs)

        if hasattr(model, "eval"):
            model = model.eval()

        # 关键改动：检查是否有 adapt_pretrained_weights 方法
        if hasattr(model, 'adapt_pretrained_weights'):
            print(f"        Model has adapt_pretrained_weights method, using adaptive loading...")
            try:
                # 使用 adapt_pretrained_weights 方法加载权重
                adaptation_result = model.adapt_pretrained_weights(model_state_dict, strict=False)

                # 打印适配信息
                if adaptation_result['size_mismatched_keys']:
                    print(f"        ✓ Successfully adapted {len(adaptation_result['size_mismatched_keys'])} dimension-mismatched layers:")
                    for key in adaptation_result['size_mismatched_keys']:
                        print(f"          - {key}")
                else:
                    print(f"        ✓ All dimensions matched, loaded normally")

            except Exception as e:
                print(f"        ✗ Adaptive loading failed: {e}")
                print(f"        Falling back to standard loading...")
                model.load_state_dict(model_state_dict, assign=True, strict=False)
        else:
            # 标准加载方式
            model.load_state_dict(model_state_dict, assign=True)

        model = model.to(dtype=torch_dtype, device=device)
        loaded_model_names.append(model_name)
        loaded_models.append(model)

    return loaded_model_names, loaded_models


class AdaptiveModelManager(BaseModelManager):
    """
    支持自动维度适配的 ModelManager

    继承自原始的 ModelManager，但重写了加载逻辑以支持：
    1. 自动检测维度不匹配
    2. 调用 adapt_pretrained_weights 进行适配
    3. 保持与原 ModelManager 的兼容性
    """

    def __init__(self, torch_dtype=torch.float16, device="cuda", use_dual_head=False):
        super().__init__(torch_dtype=torch_dtype, device=device)
        self.use_dual_head = use_dual_head
        print(f"Using AdaptiveModelManager with automatic dimension adaptation support (use_dual_head={use_dual_head})")

    def load_model_from_single_file(self, file_path="", state_dict={}, model_names=[], model_classes=[], model_resource=None):
        """
        重写加载方法以支持维度适配
        """
        print(f"Loading models from file (with adaptive support): {file_path}")
        if len(state_dict) == 0:
            state_dict = load_state_dict(file_path)

        # 使用支持适配的加载函数
        model_names, models = load_model_from_single_file_adaptive(
            state_dict, model_names, model_classes, model_resource,
            self.torch_dtype, self.device, use_dual_head=self.use_dual_head
        )

        for model_name, model in zip(model_names, models):
            self.model.append(model)
            self.model_path.append(file_path)
            self.model_name.append(model_name)

        print(f"    The following models are loaded: {model_names}.")

    def load_model(self, file_path, model_names=None, device=None, torch_dtype=None):
        """
        重写 load_model 以使用适配加载
        """
        print(f"Loading models from (adaptive): {file_path}")
        if device is None:
            device = self.device
        if torch_dtype is None:
            torch_dtype = self.torch_dtype

        if isinstance(file_path, list):
            state_dict = {}
            for path in file_path:
                state_dict.update(load_state_dict(path))
        elif os.path.isfile(file_path):
            state_dict = load_state_dict(file_path)
        else:
            state_dict = None

        for model_detector in self.model_detector:
            if model_detector.match(file_path, state_dict):
                # 使用猴子补丁临时替换 model_manager 模块中的 load_model_from_single_file 函数
                original_load_func = model_manager_module.load_model_from_single_file

                # 创建一个包装函数，自动传递 use_dual_head 参数
                def patched_load_func(state_dict, model_names, model_classes, model_resource, torch_dtype, device):
                    return load_model_from_single_file_adaptive(
                        state_dict, model_names, model_classes, model_resource,
                        torch_dtype, device, use_dual_head=self.use_dual_head
                    )

                # 临时替换全局函数
                model_manager_module.load_model_from_single_file = patched_load_func

                try:
                    model_names, models = model_detector.load(
                        file_path, state_dict,
                        device=device, torch_dtype=torch_dtype,
                        allowed_model_names=model_names, model_manager=self
                    )

                    for model_name, model in zip(model_names, models):
                        self.model.append(model)
                        self.model_path.append(file_path)
                        self.model_name.append(model_name)

                    print(f"    The following models are loaded: {model_names}.")
                finally:
                    # 恢复原始函数
                    model_manager_module.load_model_from_single_file = original_load_func

                break
        else:
            print(f"    We cannot detect the model type. No models are loaded.")


# 为了向后兼容，提供一个别名
ModelManagerWithAdaptation = AdaptiveModelManager


if __name__ == "__main__":
    print("""
    AdaptiveModelManager 使用示例:

    ```python
    from diffsynth.models.model_manager_adaptive import AdaptiveModelManager

    # 创建支持自动适配的 ModelManager
    model_manager = AdaptiveModelManager(torch_dtype=torch.bfloat16, device="cuda")

    # 正常加载模型 - 如果维度不匹配会自动适配
    model_manager.load_model(
        "path/to/pretrained_model.pth",
        device="cuda",
        torch_dtype=torch.bfloat16
    )

    # 获取模型
    dit = model_manager.fetch_model("wan_video_dit")
    ```

    适配策略:
    - patch_embedding (in_dim 变化):
      * 翻倍: 复制权重并缩放 0.5
      * 增加: 零初始化新通道
      * 减少: 截断

    - head (out_dim 变化):
      * 翻倍: 复制权重
      * 增加: 小随机值初始化
      * 减少: 截断
    """)
