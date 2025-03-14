# MiniMind LoRA训练脚本技术文档

## 1. 概述

`train_lora.py` 是 MiniMind 项目中用于实现低秩适应（Low-Rank Adaptation, LoRA）微调的训练脚本。该脚本在预训练或RLHF模型的基础上，通过LoRA技术进行参数高效微调，使模型能够适应特定领域或任务，同时保持原有模型的通用能力，并且显著减少训练参数量和计算资源需求。

## 2. 主要功能

- 加载 MiniMind 模型权重并应用 LoRA 适配器进行微调
- 实现参数高效微调，仅更新少量低秩参数
- 冻结原始模型参数，避免灾难性遗忘
- 支持分布式训练，实现单机多卡训练
- 支持梯度累积，有效扩大批次大小
- 实现混合精度训练，提高训练效率
- 支持余弦学习率调度，优化训练过程
- 提供 Weights & Biases (wandb) 集成，方便训练监控
- 定期保存 LoRA 权重，确保训练成果不丢失

## 3. LoRA 技术原理

LoRA（低秩适应）是一种参数高效微调技术，其核心思想是：

1. 冻结预训练模型的原始权重，不直接更新它们
2. 为每个需要微调的权重矩阵添加一个低秩分解的更新矩阵
3. 仅训练这些低秩矩阵，大幅减少可训练参数数量

在本实现中，LoRA 的核心结构如下：

```python
class LoRA(nn.Module):
    def __init__(self, in_features, out_features, rank):
        super().__init__()
        self.rank = rank  # LoRA的秩（rank），控制低秩矩阵的大小
        self.A = nn.Linear(in_features, rank, bias=False)  # 低秩矩阵A
        self.B = nn.Linear(rank, out_features, bias=False)  # 低秩矩阵B
        # 矩阵A高斯初始化
        self.A.weight.data.normal_(mean=0.0, std=0.02)
        # 矩阵B全0初始化
        self.B.weight.data.zero_()

    def forward(self, x):
        return self.B(self.A(x))
```

其中：
- `rank` 是低秩矩阵的秩，控制可训练参数的数量
- 矩阵 A 使用高斯分布初始化，确保训练开始时有一定的随机性
- 矩阵 B 使用零初始化，确保训练开始时 LoRA 不影响原始模型输出
- 最终的模型输出是原始权重和 LoRA 权重的和：`y = W·x + ΔW·x`，其中 `ΔW = B·A`

## 4. 主要函数说明

### 4.1 `Logger(content)`

**功能**：在分布式训练环境中只在主进程上打印日志

**参数**：
- `content`：要打印的日志内容

**处理流程**：
1. 检查是否为分布式训练环境
2. 如果不是分布式环境或者是主进程（rank=0），则打印内容

### 4.2 `get_lr(current_step, total_steps, lr)`

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

### 4.3 `train_epoch(epoch, wandb)`

**功能**：执行一个训练轮次

**参数**：
- `epoch`：当前训练轮次
- `wandb`：Weights & Biases 对象，用于记录训练指标

**处理流程**：
1. 遍历数据加载器中的每个批次
2. 将数据移至指定设备
3. 根据当前步骤计算学习率并更新优化器
4. 使用混合精度上下文执行前向传播
5. 计算交叉熵损失并应用损失掩码
6. 执行反向传播
7. 根据梯度累积步数执行优化器步骤
8. 定期记录和打印训练状态
9. 定期保存 LoRA 权重

### 4.4 `init_model(lm_config)`

**功能**：初始化并加载模型和分词器

**参数**：
- `lm_config`：语言模型配置对象

**返回值**：
- 初始化好的模型
- 对应的分词器

**处理流程**：
1. 加载分词器
2. 创建模型实例
3. 加载预训练或RLHF模型权重
4. 将模型移至指定设备
5. 返回模型和分词器

### 4.5 `init_distributed_mode()`

**功能**：初始化分布式训练环境

**处理流程**：
1. 检查是否为分布式训练环境
2. 初始化分布式进程组（使用 NCCL 后端）
3. 获取分布式训练的全局排名、本地排名和世界大小
4. 设置当前设备

### 4.6 `apply_lora(model, rank=16)`

**功能**：为模型的线性层应用 LoRA 适配器

**参数**：
- `model`：要应用 LoRA 的模型
- `rank`：LoRA 的秩，默认为 16

**处理流程**：
1. 遍历模型的所有模块
2. 对于方阵线性层（输入维度等于输出维度的层），创建 LoRA 实例
3. 将 LoRA 实例附加到线性层
4. 修改线性层的前向传播函数，使其包含 LoRA 的输出

### 4.7 `save_lora(model, path)`

**功能**：保存模型的 LoRA 权重

**参数**：
- `model`：包含 LoRA 适配器的模型
- `path`：保存路径

**处理流程**：
1. 创建一个空字典用于存储 LoRA 权重
2. 遍历模型的所有模块
3. 对于包含 LoRA 属性的模块，提取其 LoRA 权重
4. 将权重保存到指定路径

## 5. 命令行参数说明

| 参数 | 默认值 | 类型 | 说明 |
|------|--------|------|------|
| `--out_dir` | 'out' | str | 输出目录，存放模型权重文件的位置 |
| `--epochs` | 50 | int | 训练轮数 |
| `--batch_size` | 16 | int | 每个批次的样本数 |
| `--learning_rate` | 5e-5 | float | 基础学习率 |
| `--device` | 'cuda:0'/'cpu' | str | 训练设备，自动检测 CUDA 可用性 |
| `--dtype` | 'bfloat16' | str | 训练精度类型 |
| `--use_wandb` | False | bool | 是否使用 Weights & Biases 记录训练过程 |
| `--wandb_project` | 'MiniMind-LoRA-SFT' | str | Weights & Biases 项目名称 |
| `--num_workers` | 1 | int | 数据加载器的工作线程数 |
| `--ddp` | False | bool | 是否启用分布式数据并行训练 |
| `--accumulation_steps` | 1 | int | 梯度累积步数 |
| `--grad_clip` | 1.0 | float | 梯度裁剪阈值 |
| `--warmup_iters` | 0 | int | 预热迭代次数 |
| `--log_interval` | 100 | int | 日志记录间隔步数 |
| `--save_interval` | 1 | int | 模型保存间隔步数（以轮次为单位） |
| `--local_rank` | -1 | int | 本地进程排名，用于分布式训练 |
| `--dim` | 512 | int | 模型隐藏层维度 |
| `--n_layers` | 8 | int | 模型层数 |
| `--max_seq_len` | 512 | int | 最大序列长度 |
| `--use_moe` | False | bool | 是否使用混合专家模型 (MoE) |
| `--data_path` | './dataset/lora_identity.jsonl' | str | 训练数据路径 |
| `--lora_name` | 'lora_identity' | str | LoRA 权重保存名称，建议根据任务命名 |

## 6. 数据格式说明

LoRA 微调使用与 SFT 相同的数据格式，由 `SFTDataset` 类处理。训练数据应为 JSONL 格式，每行包含一个 JSON 对象，格式如下：

```json
{
  "conversations": [
    {"role": "user", "content": "请介绍一下人工智能的发展历史"},
    {"role": "assistant", "content": "人工智能的发展历史可以追溯到20世纪50年代..."}
  ]
}
```

`SFTDataset` 类会将对话转换为模型可处理的格式，并生成动态损失掩码，确保模型只在助手回复部分计算损失。

## 7. 使用示例

### 7.1 基本使用

```bash
# 使用默认配置（Small版本）进行LoRA微调
python train_lora.py
```

### 7.2 训练更大的模型

```bash
# 训练Base版本模型
python train_lora.py --dim 768 --n_layers 16
```

### 7.3 使用MoE模型

```bash
# 训练MoE版本的模型
python train_lora.py --dim 640 --use_moe True
```

### 7.4 使用分布式训练

```bash
# 使用torchrun启动分布式训练（2个GPU）
torchrun --nproc_per_node 2 train_lora.py --ddp
```

### 7.5 自定义训练配置

```bash
# 自定义批次大小、学习率和训练轮数
python train_lora.py --batch_size 32 --learning_rate 1e-4 --epochs 20
```

### 7.6 为特定领域任务训练LoRA

```bash
# 为医学领域训练LoRA
python train_lora.py --data_path ./dataset/medical_data.jsonl --lora_name lora_medical
```

## 8. 注意事项

1. LoRA 微调的学习率（默认为 5e-5）比全参数微调高得多，这是因为只更新少量参数，需要更大的学习率来有效学习。

2. LoRA 微调前需确保基础模型权重文件存在于 `./out/` 目录中，默认加载的是 RLHF 模型权重。

3. LoRA 的秩（rank）参数在 `apply_lora` 函数中默认设置为 16，这个值影响可训练参数的数量和模型的表达能力，可以根据任务复杂度和可用资源进行调整。

4. 训练数据质量对 LoRA 微调效果影响很大，建议使用与目标领域或任务高度相关的高质量数据。

5. LoRA 权重会保存在 `{args.save_dir}/lora/{args.lora_name}_{lm_config.dim}.pth` 路径下，可以通过 `--lora_name` 参数指定不同的名称以区分不同任务的 LoRA 权重。

6. 默认情况下，LoRA 只应用于方阵线性层（输入维度等于输出维度的层），这是为了减少参数量，但也可以修改 `apply_lora` 函数以应用于所有线性层。

7. 对于大规模训练，建议启用 Weights & Biases 监控，可以通过 `--use_wandb` 参数开启。

8. 使用 LoRA 微调时，模型的推理速度几乎不受影响，因为 LoRA 参数可以在推理前与原始权重合并。

## 9. 流程图

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
应用LoRA适配器到模型
  ↓
冻结原始模型参数
  ↓
收集LoRA参数
  ↓
初始化优化器（仅针对LoRA参数）
  ↓
创建数据集和数据加载器
  ↓
初始化梯度缩放器
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
  定期保存LoRA权重
```

## 10. 可扩展方法

### 10.1 添加验证集评估

可以扩展脚本以在训练过程中评估模型在验证集上的性能：

```python
def evaluate(model, val_loader, device):
    model.eval()
    total_loss = 0
    total_samples = 0
    loss_fct = nn.CrossEntropyLoss(reduction='none')
    
    with torch.no_grad():
        for X, Y, loss_mask in val_loader:
            X = X.to(device)
            Y = Y.to(device)
            loss_mask = loss_mask.to(device)
            
            res = model(X)
            loss = loss_fct(
                res.logits.view(-1, res.logits.size(-1)),
                Y.view(-1)
            ).view(Y.size())
            loss = (loss * loss_mask).sum() / loss_mask.sum()
            
            total_loss += loss.item()
            total_samples += 1
    
    model.train()
    return total_loss / total_samples
```

### 10.2 添加LoRA权重合并功能

可以添加将LoRA权重与原始模型权重合并的功能，便于部署：

```python
def merge_lora_weights(model, save_path):
    """将LoRA权重合并到原始模型中并保存"""
    # 创建模型副本
    merged_model = copy.deepcopy(model)
    
    # 合并权重
    for name, module in merged_model.named_modules():
        if hasattr(module, 'lora'):
            # 获取原始线性层权重
            original_weight = module.weight.data
            
            # 计算LoRA权重
            lora_A = module.lora.A.weight.data
            lora_B = module.lora.B.weight.data
            lora_weight = torch.matmul(lora_B.t(), lora_A.t())
            
            # 合并权重
            module.weight.data = original_weight + lora_weight
            
            # 移除LoRA属性
            delattr(module, 'lora')
            # 恢复原始前向传播函数
            module.forward = types.MethodType(nn.Linear.forward, module)
    
    # 保存合并后的模型
    torch.save(merged_model.state_dict(), save_path)
    Logger(f"Merged model saved to {save_path}")
    
    return merged_model
```

### 10.3 实现更灵活的LoRA应用策略

可以实现更灵活的LoRA应用策略，针对不同类型的层使用不同的秩：

```python
def apply_lora_advanced(model, config):
    """高级LoRA应用函数，支持针对不同层类型使用不同的秩"""
    # 配置示例：{'attention': 8, 'mlp': 16, 'default': 4}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # 确定应用的秩
            rank = config.get('default', 16)
            if 'attention' in name:
                rank = config.get('attention', rank)
            elif 'mlp' in name:
                rank = config.get('mlp', rank)
            
            # 创建并应用LoRA
            lora = LoRA(module.in_features, module.out_features, rank=rank).to(model.device)
            setattr(module, "lora", lora)
            original_forward = module.forward
            
            # 显式绑定
            def forward_with_lora(x, layer1=original_forward, layer2=lora):
                return layer1(x) + layer2(x)
            
            module.forward = forward_with_lora
```

### 10.4 添加LoRA权重可视化功能

可以添加LoRA权重可视化功能，帮助理解模型学习到的内容：

```python
def visualize_lora_weights(model, save_dir):
    """可视化LoRA权重"""
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    
    os.makedirs(save_dir, exist_ok=True)
    
    for name, module in model.named_modules():
        if hasattr(module, 'lora'):
            # 获取LoRA权重
            lora_A = module.lora.A.weight.data.cpu().numpy()
            lora_B = module.lora.B.weight.data.cpu().numpy()
            
            # 计算合成权重
            lora_weight = np.matmul(lora_B.T, lora_A.T)
            
            # 绘制热力图
            plt.figure(figsize=(10, 8))
            plt.imshow(lora_weight, cmap='viridis')
            plt.colorbar()
            plt.title(f'LoRA Weight for {name}')
            plt.savefig(os.path.join(save_dir, f'{name.replace(".", "_")}.png'))
            plt.close()
```

## 11. 开发者信息

本脚本是MiniMind项目的一部分，用于实现参数高效的模型微调。LoRA技术通过只更新少量低秩参数，大幅减少了微调所需的计算资源和存储空间，同时保持了与全参数微调相当的性能。这使得在有限资源条件下对大型语言模型进行领域适应成为可能，为模型的实际应用提供了更灵活的解决方案。