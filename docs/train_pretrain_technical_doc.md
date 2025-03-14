# MiniMind 预训练脚本技术文档

## 1. 概述

`train_pretrain.py` 是 MiniMind 项目中用于从零开始训练语言模型的预训练脚本。该脚本支持多种模型配置和训练模式，包括标准 Transformer 结构和混合专家模型 (MoE) 结构，并提供了分布式训练支持，使用 PyTorch 的 DistributedDataParallel 进行多卡训练。

## 2. 主要功能

- 支持不同规模的 MiniMind 模型训练（通过调整维度和层数）
- 支持混合专家模型 (MoE) 结构的训练
- 提供分布式训练支持，实现单机多卡训练
- 支持梯度累积，有效扩大批次大小
- 实现混合精度训练，提高训练效率
- 支持余弦学习率调度，优化训练过程
- 提供 Weights & Biases (wandb) 集成，方便训练监控
- 定期保存模型权重，确保训练成果不丢失

## 3. 主要函数说明

### 3.1 `Logger(content)`

**功能**：在分布式训练环境中只在主进程上打印日志

**参数**：
- `content`：要打印的日志内容

**处理流程**：
1. 检查是否为分布式训练环境
2. 如果不是分布式环境或者是主进程（rank=0），则打印内容

### 3.2 `get_lr(current_step, total_steps, lr)`

**功能**：计算当前步骤的学习率，实现余弦衰减调度

**参数**：
- `current_step`：当前训练步骤
- `total_steps`：总训练步骤数
- `lr`：基础学习率

**返回值**：
- 当前步骤应用的学习率值

**计算公式**：
```
lr / 10 + 0.5 * lr * (1 + cos(π * current_step / total_steps))
```

### 3.3 `train_epoch(epoch, wandb)`

**功能**：执行一个训练轮次

**参数**：
- `epoch`：当前训练轮次
- `wandb`：Weights & Biases 对象，用于记录训练指标

**处理流程**：
1. 初始化交叉熵损失函数
2. 遍历数据加载器中的每个批次
3. 将数据移至指定设备
4. 根据当前步骤计算学习率并更新优化器
5. 使用混合精度上下文执行前向传播
6. 计算损失（包括辅助损失）并进行归一化
7. 执行反向传播
8. 根据梯度累积步数执行优化器步骤
9. 定期记录和打印训练状态
10. 定期保存模型权重

### 3.4 `init_model(lm_config)`

**功能**：初始化并加载模型和分词器

**参数**：
- `lm_config`：语言模型配置对象

**返回值**：
- 初始化好的模型
- 对应的分词器

**处理流程**：
1. 加载分词器
2. 创建 MiniMindLM 模型实例并移至指定设备
3. 打印模型参数量
4. 返回模型和分词器

### 3.5 `init_distributed_mode()`

**功能**：初始化分布式训练环境

**处理流程**：
1. 检查是否为分布式训练环境
2. 初始化分布式进程组（使用 NCCL 后端）
3. 获取分布式训练的全局排名、本地排名和世界大小
4. 设置当前设备

## 4. 命令行参数说明

| 参数 | 默认值 | 类型 | 说明 |
|------|--------|------|------|
| `--out_dir` | 'out' | str | 输出目录，存放模型权重文件的位置 |
| `--epochs` | 1 | int | 训练轮数 |
| `--batch_size` | 32 | int | 每个批次的样本数 |
| `--learning_rate` | 5e-4 | float | 基础学习率 |
| `--device` | 'cuda:0'/'cpu' | str | 训练设备，自动检测 CUDA 可用性 |
| `--dtype` | 'bfloat16' | str | 训练精度类型 |
| `--use_wandb` | False | bool | 是否使用 Weights & Biases 记录训练过程 |
| `--wandb_project` | 'MiniMind-Pretrain' | str | Weights & Biases 项目名称 |
| `--num_workers` | 1 | int | 数据加载器的工作线程数 |
| `--ddp` | False | bool | 是否启用分布式数据并行训练 |
| `--accumulation_steps` | 8 | int | 梯度累积步数 |
| `--grad_clip` | 1.0 | float | 梯度裁剪阈值 |
| `--warmup_iters` | 0 | int | 预热迭代次数 |
| `--log_interval` | 100 | int | 日志记录间隔步数 |
| `--save_interval` | 100 | int | 模型保存间隔步数 |
| `--local_rank` | -1 | int | 本地进程排名，用于分布式训练 |
| `--dim` | 512 | int | 模型隐藏层维度 |
| `--n_layers` | 8 | int | 模型层数 |
| `--max_seq_len` | 512 | int | 最大序列长度 |
| `--use_moe` | False | bool | 是否使用混合专家模型 (MoE) |
| `--data_path` | './dataset/pretrain_hq.jsonl' | str | 预训练数据路径 |

## 5. 模型配置说明

脚本支持以下预设的模型配置：

- **MiniMind-Small (26M)**：dim=512, n_layers=8
- **MiniMind-Base (104M)**：dim=768, n_layers=16
- **MiniMind-MoE (145M)**：dim=640, n_layers=8, use_moe=True

## 6. 使用示例

### 6.1 基本使用

```bash
# 使用默认配置（Small版本）训练模型
python train_pretrain.py
```

### 6.2 训练更大的模型

```bash
# 训练Base版本模型
python train_pretrain.py --dim 768 --n_layers 16
```

### 6.3 使用MoE模型

```bash
# 训练MoE版本的模型
python train_pretrain.py --dim 640 --use_moe True
```

### 6.4 使用分布式训练

```bash
# 使用torchrun启动分布式训练（2个GPU）
torchrun --nproc_per_node 2 train_pretrain.py
```

### 6.5 自定义训练配置

```bash
# 自定义批次大小、学习率和训练轮数
python train_pretrain.py --batch_size 64 --learning_rate 1e-4 --epochs 3
```

## 7. 注意事项

1. 预训练脚本默认使用 `bfloat16` 精度进行混合精度训练，如果硬件不支持，可以修改为 `float16` 或 `float32`。

2. 梯度累积步数 `accumulation_steps` 默认为8，可以根据显存大小调整。较小的显存可以增加累积步数，较大的显存可以减少累积步数并增加批次大小。

3. 分布式训练时，需要使用 `torchrun` 命令启动，而不是直接使用 `python` 命令。

4. 对于大规模训练，建议启用 Weights & Biases 监控，可以通过 `--use_wandb` 参数开启。

5. 模型权重会定期保存在 `out_dir` 指定的目录中，文件名格式为 `pretrain_{dim}{_moe}.pth`。

6. 预训练数据应为 JSONL 格式，每行包含一个 JSON 对象，由 `PretrainDataset` 类处理。

7. 如果要以最快速度实现预训练，可以将 `epochs` 设置为1；如果要充分利用有限的数据，建议设置为2~6个轮次。

## 8. 流程图

```
初始化
  ↓
解析命令行参数
  ↓
创建模型配置
  ↓
初始化分布式环境（如果启用）
  ↓
初始化Weights & Biases（如果启用）
  ↓
加载模型和分词器
  ↓
创建数据集和数据加载器
  ↓
初始化优化器和梯度缩放器
  ↓
包装模型为DistributedDataParallel（如果启用分布式训练）
  ↓
循环训练每个轮次
  ↓
  计算学习率
  ↓
  前向传播
  ↓
  计算损失
  ↓
  反向传播
  ↓
  梯度累积和裁剪
  ↓
  优化器步骤
  ↓
  记录训练状态
  ↓
  定期保存模型
```

## 9. 可扩展方法

### 9.1 添加预训练数据处理功能

可以扩展脚本以支持更多类型的预训练数据：

```python
# 在model/dataset.py中扩展PretrainDataset类
class EnhancedPretrainDataset(PretrainDataset):
    def __init__(self, data_path, tokenizer, max_length=512, data_format="jsonl"):
        self.data_format = data_format
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        if data_format == "jsonl":
            # 原有的JSONL处理逻辑
            self.data = []
            with open(data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    self.data.append(json.loads(line))
        elif data_format == "txt":
            # 处理纯文本文件
            self.data = []
            with open(data_path, 'r', encoding='utf-8') as f:
                text = f.read()
                # 按段落分割
                paragraphs = text.split('\n\n')
                for para in paragraphs:
                    if para.strip():
                        self.data.append({"text": para.strip()})
        elif data_format == "csv":
            # 处理CSV文件
            import pandas as pd
            df = pd.read_csv(data_path)
            text_column = df.columns[0]  # 假设第一列是文本
            self.data = [{"text": row[text_column]} for _, row in df.iterrows()]
```

在 `main()` 函数中添加相应的命令行参数：

```python
parser.add_argument("--data_format", type=str, default="jsonl", 
                    choices=["jsonl", "txt", "csv"],
                    help="预训练数据格式")
```

### 9.2 添加模型检查点恢复功能

可以添加从检查点恢复训练的功能：

```python
def load_checkpoint(checkpoint_path, model, optimizer, scaler):
    """从检查点恢复训练状态"""
    if os.path.exists(checkpoint_path):
        Logger(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=args.device)
        
        # 加载模型权重
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # 加载优化器状态
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # 加载梯度缩放器状态
        if 'scaler_state_dict' in checkpoint and scaler is not None:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        # 返回上次训练的轮次和步骤
        epoch = checkpoint.get('epoch', 0)
        step = checkpoint.get('step', 0)
        return epoch, step
    return 0, 0

def save_checkpoint(checkpoint_path, model, optimizer, scaler, epoch, step):
    """保存训练检查点"""
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_state_dict = model.module.state_dict()
    else:
        model_state_dict = model.state_dict()
    
    checkpoint = {
        'model_state_dict': model_state_dict,
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'step': step
    }
    
    if scaler is not None:
        checkpoint['scaler_state_dict'] = scaler.state_dict()
    
    torch.save(checkpoint, checkpoint_path)
    Logger(f"Checkpoint saved to {checkpoint_path}")
```

在 `main()` 函数中添加相应的命令行参数：

```python
parser.add_argument("--resume", action="store_true", help="从检查点恢复训练")
parser.add_argument("--checkpoint_path", type=str, default=None, help="检查点文件路径")
```

### 9.3 优化学习率调度

可以扩展学习率调度功能，添加预热和多阶段衰减：

```python
def get_advanced_lr(current_step, total_steps, base_lr, warmup_steps=0, decay_style="cosine"):
    """高级学习率调度函数"""
    # 预热阶段
    if current_step < warmup_steps:
        return base_lr * (current_step / max(1, warmup_steps))
    
    # 预热后阶段
    if decay_style == "cosine":
        # 余弦衰减
        decay_ratio = 0.1  # 最终学习率为初始学习率的10%
        decay_steps = total_steps - warmup_steps
        current_decay_step = current_step - warmup_steps
        decay_factor = 0.5 * (1.0 + math.cos(math.pi * current_decay_step / decay_steps))
        return decay_ratio * base_lr + (base_lr - decay_ratio * base_lr) * decay_factor
    elif decay_style == "linear":
        # 线性衰减
        decay_steps = total_steps - warmup_steps
        current_decay_step = current_step - warmup_steps
        return base_lr * (1.0 - current_decay_step / decay_steps)
    elif decay_style == "step":
        # 阶梯式衰减
        decay_ratio = 0.1
        decay_at = [0.5, 0.75]  # 在50%和75%的训练步骤处衰减
        decay_steps = total_steps - warmup_steps
        current_decay_step = current_step - warmup_steps
        decay_factor = 1.0
        for ratio in decay_at:
            if current_decay_step >= ratio * decay_steps:
                decay_factor *= decay_ratio
        return base_lr * decay_factor
    else:
        return base_lr
```

在 `main()` 函数中添加相应的命令行参数：

```python
parser.add_argument("--warmup_steps", type=int, default=0, help="学习率预热步数")
parser.add_argument("--decay_style", type=str, default="cosine", 
                    choices=["cosine", "linear", "step"], help="学习率衰减方式")
```

### 9.4 分布式训练最佳实践

为了优化分布式训练性能，可以添加以下功能：

```python
def setup_distributed_training():
    """设置分布式训练的最佳实践"""
    # 设置NCCL环境变量以优化性能
    os.environ["NCCL_DEBUG"] = "INFO"  # 可选：设置为WARN减少日志输出
    os.environ["NCCL_IB_DISABLE"] = "0"  # 启用InfiniBand支持（如果可用）
    
    # 设置CUDA相关环境变量
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # 确保CUDA设备顺序一致
    
    # 设置PyTorch分布式相关参数
    torch.backends.cudnn.benchmark = True  # 启用cuDNN基准测试以提高性能
    
    # 对于大模型，可以启用以下选项以减少内存使用
    if args.dim >= 768:
        torch.backends.cuda.matmul.allow_tf32 = True  # 允许使用TF32精度（对于Ampere及以上架构）
        torch.backends.cudnn.allow_tf32 = True  # 允许cuDNN使用TF32精度
```

### 9.5 添加训练监控和可视化

除了Weights & Biases，还可以添加更多的训练监控和可视化功能：

```python
def setup_monitoring(args):
    """设置训练监控和可视化"""
    monitoring = {}
    
    if args.use_wandb and (not ddp or dist.get_rank() == 0):
        import wandb
        wandb.init(project=args.wandb_project, name=args.wandb_run_name)
        monitoring["wandb"] = wandb
    else:
        monitoring["wandb"] = None
    
    if args.use_tensorboard and (not ddp or dist.get_rank() == 0):
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=os.path.join(args.out_dir, "tensorboard_logs"))
        monitoring["tensorboard"] = writer
    else:
        monitoring["tensorboard"] = None
    
    # 设置训练进度条
    if not ddp or dist.get_rank() == 0:
        try:
            from tqdm import tqdm
            monitoring["use_tqdm"] = True
        except ImportError:
            monitoring["use_tqdm"] = False
    else:
        monitoring["use_tqdm"] = False
    
    return monitoring

def log_metrics(monitoring, metrics, step):
    """记录训练指标"""
    if monitoring["wandb"] is not None:
        monitoring["wandb"].log(metrics, step=step)
    
    if monitoring["tensorboard"] is not None:
        for key, value in metrics.items():
            monitoring["tensorboard"].add_scalar(key, value, step)
```

在 `main()` 函数中添加相应的命令行参数：

```python
parser.add_argument("--use_tensorboard", action="store_true", help="使用TensorBoard记录训练过程")
```

## 10. 开发者信息

本脚本是MiniMind项目的一部分，用于从零开始训练语言模型的预训练阶段。预训练是构建语言模型的基础步骤，通过大规模文本数据训练模型学习词语接龙能力，让模型参数中充满知识的"墨水"。