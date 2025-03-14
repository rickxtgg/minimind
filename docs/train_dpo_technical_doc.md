# MiniMind DPO训练脚本技术文档

## 1. 概述

`train_dpo.py` 是 MiniMind 项目中用于实现直接偏好优化（Direct Preference Optimization, DPO）的训练脚本。该脚本在监督微调（SFT）模型的基础上，通过人类偏好数据进一步优化模型，使模型生成的内容更符合人类偏好，从而提高模型的实用性和安全性。

## 2. 主要功能

- 加载经过监督微调的 MiniMind 模型权重并进行 DPO 训练
- 实现 DPO 算法，通过偏好对比学习优化模型输出
- 使用参考模型（Reference Model）稳定训练过程
- 支持分布式训练，实现单机多卡训练
- 支持梯度累积，有效扩大批次大小
- 实现混合精度训练，提高训练效率
- 支持余弦学习率调度，优化训练过程
- 提供 Weights & Biases (wandb) 集成，方便训练监控
- 定期保存模型权重，确保训练成果不丢失

## 3. DPO 算法原理

DPO（直接偏好优化）是一种基于人类偏好数据优化语言模型的算法，其核心思想是：

1. 使用一个固定的参考模型（通常是 SFT 模型）作为基准
2. 训练一个策略模型，使其在人类偏好的回答上的概率比在非偏好回答上的概率更高
3. 同时相对于参考模型保持适当的 KL 散度约束，防止过度偏离原始模型

在本实现中，DPO 损失函数的计算如下：

```python
def dpo_loss(ref_probs, probs, beta):
    # 计算每个样本的平均概率
    ref_probs = ref_probs.mean(dim=1)
    probs = probs.mean(dim=1)

    # 将 chosen 和 rejected 数据分开
    batch_size = ref_probs.shape[0]
    chosen_ref_probs = ref_probs[:batch_size // 2]
    reject_ref_probs = ref_probs[batch_size // 2:]
    chosen_probs = probs[:batch_size // 2]
    reject_probs = probs[batch_size // 2:]

    pi_logratios = chosen_probs - reject_probs
    ref_logratios = chosen_ref_probs - reject_ref_probs
    logits = pi_logratios - ref_logratios
    loss = -F.logsigmoid(beta * logits)
    return loss.mean()
```

其中：
- `beta` 是温度参数，控制 KL 散度约束的强度
- `pi_logratios` 表示策略模型在偏好和非偏好回答上的对数概率比
- `ref_logratios` 表示参考模型在偏好和非偏好回答上的对数概率比
- 最终的损失函数鼓励策略模型在偏好回答上的概率比参考模型更高

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

### 4.3 `logits_to_probs(logits, labels)`

**功能**：将模型输出的 logits 转换为对应标签的概率

**参数**：
- `logits`：模型输出的 logits，形状为 (batch_size, seq_len, vocab_size)
- `labels`：真实标签，形状为 (batch_size, seq_len)

**返回值**：
- 对应标签的概率，形状为 (batch_size, seq_len)

**处理流程**：
1. 对 logits 在词汇表维度上应用 log_softmax
2. 使用 gather 操作提取对应标签的概率

### 4.4 `dpo_loss(ref_probs, probs, beta)`

**功能**：计算 DPO 损失

**参数**：
- `ref_probs`：参考模型的概率，形状为 (batch_size, seq_len)
- `probs`：策略模型的概率，形状为 (batch_size, seq_len)
- `beta`：温度参数，控制 KL 散度约束的强度

**返回值**：
- DPO 损失值

**处理流程**：
1. 计算每个样本的平均概率
2. 将数据分为偏好（chosen）和非偏好（rejected）两部分
3. 计算策略模型和参考模型在偏好和非偏好回答上的对数概率比
4. 计算最终的 DPO 损失

### 4.5 `train_epoch(epoch, wandb)`

**功能**：执行一个训练轮次

**参数**：
- `epoch`：当前训练轮次
- `wandb`：Weights & Biases 对象，用于记录训练指标

**处理流程**：
1. 遍历数据加载器中的每个批次
2. 将数据移至指定设备
3. 根据当前步骤计算学习率并更新优化器
4. 使用混合精度上下文执行前向传播
5. 计算参考模型和策略模型的概率
6. 计算 DPO 损失并进行归一化
7. 执行反向传播
8. 根据梯度累积步数执行优化器步骤
9. 定期记录和打印训练状态
10. 定期保存模型权重

### 4.6 `init_model(lm_config)`

**功能**：初始化并加载模型和分词器

**参数**：
- `lm_config`：语言模型配置对象

**返回值**：
- 初始化好的策略模型
- 初始化好的参考模型
- 对应的分词器

**处理流程**：
1. 加载分词器
2. 创建策略模型实例
3. 加载 SFT 模型权重
4. 创建参考模型实例并加载相同的权重
5. 将参考模型设置为评估模式并冻结参数
6. 打印模型参数量
7. 将模型移至指定设备
8. 返回策略模型、参考模型和分词器

### 4.7 `init_distributed_mode()`

**功能**：初始化分布式训练环境

**处理流程**：
1. 检查是否为分布式训练环境
2. 初始化分布式进程组（使用 NCCL 后端）
3. 获取分布式训练的全局排名、本地排名和世界大小
4. 设置当前设备

## 5. 命令行参数说明

| 参数 | 默认值 | 类型 | 说明 |
|------|--------|------|------|
| `--out_dir` | 'out' | str | 输出目录，存放模型权重文件的位置 |
| `--epochs` | 2 | int | 训练轮数 |
| `--batch_size` | 8 | int | 每个批次的样本数 |
| `--learning_rate` | 1e-8 | float | 基础学习率 |
| `--device` | 'cuda:0'/'cpu' | str | 训练设备，自动检测 CUDA 可用性 |
| `--dtype` | 'bfloat16' | str | 训练精度类型 |
| `--use_wandb` | False | bool | 是否使用 Weights & Biases 记录训练过程 |
| `--wandb_project` | 'MiniMind-RLHF-SFT' | str | Weights & Biases 项目名称 |
| `--num_workers` | 1 | int | 数据加载器的工作线程数 |
| `--ddp` | False | bool | 是否启用分布式数据并行训练 |
| `--accumulation_steps` | 1 | int | 梯度累积步数 |
| `--grad_clip` | 1.0 | float | 梯度裁剪阈值 |
| `--warmup_iters` | 0 | int | 预热迭代次数 |
| `--log_interval` | 100 | int | 日志记录间隔步数 |
| `--save_interval` | 100 | int | 模型保存间隔步数 |
| `--local_rank` | -1 | int | 本地进程排名，用于分布式训练 |
| `--dim` | 512 | int | 模型隐藏层维度 |
| `--n_layers` | 8 | int | 模型层数 |
| `--max_seq_len` | 3000 | int | 最大序列长度 |
| `--use_moe` | False | bool | 是否使用混合专家模型 (MoE) |
| `--data_path` | './dataset/dpo.jsonl' | str | DPO 训练数据路径 |

## 6. 数据格式说明

DPO 训练数据应为 JSONL 格式，每行包含一个 JSON 对象，由 `DPODataset` 类处理。每个 JSON 对象应包含 `chosen` 和 `rejected` 字段，分别表示人类偏好的回答和非偏好的回答，格式如下：

```json
{
  "chosen": [
    {"role": "user", "content": "请解释量子力学的基本原理"},
    {"role": "assistant", "content": "量子力学是描述微观粒子行为的物理理论..."}
  ],
  "rejected": [
    {"role": "user", "content": "请解释量子力学的基本原理"},
    {"role": "assistant", "content": "量子力学是一种复杂的理论，普通人很难理解..."}
  ]
}
```

`DPODataset` 类会将对话转换为 ChatML 格式，并生成动态损失掩码，确保模型只在助手回复部分计算损失。

## 7. 使用示例

### 7.1 基本使用

```bash
# 使用默认配置（Small版本）进行DPO训练
python train_dpo.py
```

### 7.2 训练更大的模型

```bash
# 训练Base版本模型
python train_dpo.py --dim 768 --n_layers 16
```

### 7.3 使用MoE模型

```bash
# 训练MoE版本的模型
python train_dpo.py --dim 640 --use_moe True
```

### 7.4 使用分布式训练

```bash
# 使用torchrun启动分布式训练（2个GPU）
torchrun --nproc_per_node 2 train_dpo.py --ddp
```

### 7.5 自定义训练配置

```bash
# 自定义批次大小、学习率和训练轮数
python train_dpo.py --batch_size 16 --learning_rate 5e-9 --epochs 3
```

## 8. 注意事项

1. DPO 训练需要非常小的学习率（默认为 1e-8），否则容易导致模型遗忘或性能下降。代码中的注释也提到："sft阶段学习率为 「5e-6」->「5e-7」长度512，建议离线正负样本「概率」偏好对齐阶段lr <=「1e-8」长度3000，否则很容易遗忘训坏"。

2. DPO 训练前需确保已完成监督微调（SFT）阶段，并且 SFT 模型权重文件存在于 `./out/` 目录中。

3. 训练数据质量对 DPO 效果影响很大，应确保偏好数据中的 chosen 和 rejected 回答有明显的质量差异，但内容相关性要高。

4. 参考模型（Reference Model）在训练过程中保持冻结状态，不更新参数，这对于 DPO 训练的稳定性至关重要。

5. 默认的序列长度（3000）比 SFT 阶段（通常为 512）要长得多，这是为了捕获更完整的对话上下文，但也会增加显存需求。

6. 模型权重会定期保存在 `out_dir` 指定的目录中，文件名格式为 `rlhf_{dim}{_moe}.pth`。

7. 对于大规模训练，建议启用 Weights & Biases 监控，可以通过 `--use_wandb` 参数开启。

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
加载SFT模型和分词器
  ↓
创建参考模型并冻结参数
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
  前向传播（参考模型和策略模型）
  ↓
  计算DPO损失
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

## 10. 可扩展方法

### 10.1 添加验证集评估

可以扩展脚本以在训练过程中评估模型在验证集上的性能：

```python
def evaluate(model, ref_model, val_loader, device):
    model.eval()
    total_loss = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch in val_loader:
            x_chosen = batch['x_chosen'].to(device)
            x_rejected = batch['x_rejected'].to(device)
            y_chosen = batch['y_chosen'].to(device)
            y_rejected = batch['y_rejected'].to(device)
            mask_chosen = batch['mask_chosen'].to(device)
            mask_rejected = batch['mask_rejected'].to(device)
            x = torch.cat([x_chosen, x_rejected], dim=0)
            y = torch.cat([y_chosen, y_rejected], dim=0)
            mask = torch.cat([mask_chosen, mask_rejected], dim=0)
            
            ref_outputs = ref_model(x)
            ref_logits = ref_outputs.logits
            ref_probs = logits_to_probs(ref_logits, y)
            ref_probs = ref_probs * mask
            
            outputs = model(x)
            logits = outputs.logits
            probs = logits_to_probs(logits, y)
            probs = probs * mask
            
            loss = dpo_loss(ref_probs, probs, beta=0.1)
            
            total_loss += loss.item()
            total_samples += 1
    
    model.train()
    return total_loss / total_samples
```

### 10.2 添加模型检查点恢复功能

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
```

### 10.3 实现更灵活的 beta 参数调整

可以实现动态调整 beta 参数的功能，以在训练过程中平衡偏好学习和 KL 约束：

```python
def get_beta(current_step, total_steps, min_beta=0.1, max_beta=0.5):
    """动态调整beta参数"""
    # 训练初期使用较小的beta，减少对参考模型的偏离
    # 训练后期增大beta，加强偏好学习
    progress = current_step / total_steps
    return min_beta + (max_beta - min_beta) * progress
```

### 10.4 添加人类评估接口

可以添加一个简单的评估接口，用于人类评估模型生成的内容：

```python
def generate_for_evaluation(model, tokenizer, prompts, max_length=1024):
    """生成用于人类评估的回答"""
    model.eval()
    results = []
    
    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        chat_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        inputs = tokenizer(chat_prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_length=max_length,
                temperature=0.7,
                top_p=0.9,
                do_sample=True
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        results.append({"prompt": prompt, "response": response})
    
    return results
```

## 11. 开发者信息

本脚本是MiniMind项目的一部分，用于实现基于人类偏好的模型优化。DPO是RLHF（基于人类反馈的强化学习）的一种简化实现，它避免了传统RLHF中PPO算法的复杂性，同时保持了类似的效果。通过DPO训练，模型可以学习生成更符合人类偏好的回答，提高回答的有用性、真实性和安全性。