import torch
from diffusers import StableDiffusionPipeline
from transformers import Trainer, TrainingArguments
from datasets import load_dataset
from torchvision import transforms

# 1. **加载预训练的Stable Diffusion模型**
model_id = "CompVis/stable-diffusion-v1-4-original"
pipe = StableDiffusionPipeline.from_pretrained(model_id)
pipe.to("cuda" if torch.cuda.is_available() else "cpu")

# 2. **准备数据集（以图像文本数据集为例）**
# 假设你有一个图像和对应文本的数据集
# 使用 datasets 库加载数据集（或者你也可以用其他方式加载数据）
dataset = load_dataset("path_to_your_dataset")

# 数据预处理函数
def preprocess_image(image):
    # 例如，标准化并调整大小
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # 根据预训练模型的需求进行调整
    ])
    return transform(image)

# 处理数据集
def process_data(example):
    example["pixel_values"] = preprocess_image(example["image"])  # 预处理图像
    return example

dataset = dataset.map(process_data)

# 3. **配置微调训练参数**
training_args = TrainingArguments(
    output_dir="./results",            # 输出目录
    per_device_train_batch_size=4,     # 每个设备上的训练批次大小
    num_train_epochs=3,                # 训练轮数
    logging_dir="./logs",              # 日志目录
    save_steps=10_000,                 # 保存模型的步数
    save_total_limit=2,                # 最大保存模型数量
    gradient_accumulation_steps=2,     # 梯度累积步数
    report_to="tensorboard",           # 使用tensorboard进行监控
    logging_steps=500,                 # 每500步记录一次日志
)

# 4. **设置Trainer进行训练**
# 使用 Trainer 进行训练，指定模型、训练数据集、评估数据集和训练参数
trainer = Trainer(
    model=pipe,                        # 预训练的模型
    args=training_args,                # 训练参数
    train_dataset=dataset["train"],    # 训练数据集
    eval_dataset=dataset["test"],      # 测试数据集
)

# 5. **开始微调训练**
trainer.train()

# 6. **保存微调后的模型**
trainer.save_model("./fine_tuned_model")

# 7. **加载微调后的模型**
pipe = StableDiffusionPipeline.from_pretrained("./fine_tuned_model")
pipe.to("cuda" if torch.cuda.is_available() else "cpu")

# 8. **生成图像**
prompt = "A futuristic city with flying cars"
generated_image = pipe(prompt).images[0]

# 9. **显示生成的图像**
generated_image.show()

# 10. **监控训练过程（可选）**
# 启用 TensorBoard 监控
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(log_dir="./logs")
# 在训练过程中，你可以在每个step添加日志：
# writer.add_scalar('Loss/train', loss, global_step)
# 或者记录训练中的其他指标
