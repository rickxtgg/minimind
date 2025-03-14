# MiniMind 监督微调脚本技术文档

## 1. 概述

`train_full_sft.py` 是 MiniMind 项目中用于对预训练模型进行监督微调（Supervised Fine-Tuning, SFT）的脚本。该脚本在预训练模型的基础上，通过高质量的指令-回复数据对模型进行微调，使模型学习遵循人类指令的能力，从而将预训练模型转变为具有对话能力的语言模型。

## 2. 主要功能

- 加载预训练的 MiniMind 模型权重并进行监督微调
- 支持标准 Transformer 结构和混合专家模型 (MoE) 结构
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
2. 创建 MiniMindLM 模型实例
3. 加载预训练模型权重
4. 打印模型参数量
5. 将模型移至指定设备
6. 返回模型和分词器

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
| `--learning_rate` | 5e-5 | float | 基础学习率 |
| `--device` | 'cuda:0'/'cpu' | str | 训练设备，自动检测 CUDA 可用性 |
| `--dtype` | 'bfloat16' | str | 训练精度类型 |
| `--use_wandb` | False | bool | 是否使用 Weights & Biases 记录训练过程 |
| `--wandb_project` | 'MiniMind-Full-SFT' | str | Weights & Biases 项目名称 |
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
| `--max_seq_len` | 512 | int | 最大序列长度 |
| `--use_moe` | False | bool | 是否使用混合专家模型 (MoE) |
| `--data_path` | './dataset/sft_mini_512.jsonl' | str | 微调数据路径 |

## 5. 模型配置说明

脚本支持以下预设的模型配置：

- **MiniMind-Small (26M)**：dim=512, n_layers=8
- **MiniMind-Base (104M)**：dim=768, n_layers=16
- **MiniMind-MoE (145M)**：dim=640, n_layers=8, use_moe=True

## 6. 数据格式说明

微调数据应为 JSONL 格式，每行包含一个 JSON 对象，由 `SFTDataset` 类处理。每个 JSON 对象应包含 `conversations` 字段，该字段是一个对话列表，格式如下：

```json
{
  "conversations": [
    {"role": "user", "content": "你好，请介绍一下自己"},
    {"role": "assistant", "content": "你好！我是MiniMind，一个由..."}
  ]
}
```

`SFTDataset` 类会将对话转换为 ChatML 格式，并生成动态损失掩码，确保模型只在助手回复部分计算损失。

## 7. 使用示例

### 7.1 基本使用

```bash
# 使用默认配置（Small版本）微调模型
python train_full_sft.py
```

### 7.2 微调更大的模型

```bash
# 微调Base版本模型
python train_full_sft.py --dim 768 --n_layers 16
```

### 7.3 使用MoE模型

```bash
# 微调MoE版本的模型
python train_full_sft.py --dim 640 --use_moe True
```

### 7.4 使用分布式训练

```bash
# 使用torchrun启动分布式训练（2个GPU）
torchrun --nproc_per_node 2 train_full_sft.py --ddp
```

### 7.5 自定义训练配置

```bash
# 自定义批次大小、学习率和训练轮数
python train_full_sft.py --batch_size 64 --learning_rate 1e-5 --epochs 3
```

## 8. 注意事项

1. 微调脚本默认使用 `bfloat16` 精度进行混合精度训练，如果硬件不支持，可以修改为 `float16` 或 `float32`。

2. 微调前需确保已完成预训练阶段，并且预训练模型权重文件存在于 `./out/` 目录中。

3. 梯度累积步数 `accumulation_steps` 默认为1，可以根据显存大小调整。较小的显存可以增加累积步数，较大的显存可以减少累积步数并增加批次大小。

4. 分布式训练时，需要使用 `torchrun` 命令启动，而不是直接使用 `python` 命令。

5. 对于大规模训练，建议启用 Weights & Biases 监控，可以通过 `--use_wandb` 参数开启。

6. 模型权重会定期保存在 `out_dir` 指定的目录中，文件名格式为 `full_sft_{dim}{_moe}.pth`。

7. 微调数据质量对最终模型性能影响很大，建议使用高质量的指令-回复对。

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
加载预训练模型和分词器
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

## 10. 可扩展方法

### 10.1 添加验证集评估

可以扩展脚本以在训练过程中评估模型在验证集上的性能：

```python
def evaluate(model, val_loader, device):
    model.eval()
    total_loss = 0
    total_tokens = 0
    loss_fct = nn.CrossEntropyLoss(reduction='none')
    
    with torch.no_grad():
        for X, Y, loss_mask in val_loader:
            X = X.to(device)
            Y = Y.to(device)
            loss_mask = loss_mask.to(device)
            
            with torch.cuda.amp.autocast():
                res = model(X)
                loss = loss_fct(
                    res.logits.view(-1, res.logits.size(-1)),
                    Y.view(-1)
                ).view(Y.size())
                
                loss = (loss * loss_mask).sum()
                total_loss += loss.item()
                total_tokens += loss_mask.sum().item()
    
    model.train()
    return total_loss / total_tokens
```

在 `train_epoch` 函数中添加验证逻辑：

```python
if (step + 1) % args.eval_interval == 0 and (not ddp or dist.get_rank() == 0):
    val_loss = evaluate(model, val_loader, args.device)
    Logger(f'Validation loss: {val_loss:.4f}')
    if wandb is not None:
        wandb.log({"val_loss": val_loss})
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

### 10.3 优化学习率调度

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

### 10.4 添加数据增强功能

可以扩展 `SFTDataset` 类以支持数据增强：

```python
class EnhancedSFTDataset(SFTDataset):
    def __init__(self, jsonl_path, tokenizer, max_length=1024, augment=False):
        super().__init__(jsonl_path, tokenizer, max_length)
        self.augment = augment
        
    def __getitem__(self, index):
        sample = self.samples[index]
        
        # 数据增强
        if self.augment and random.random() < 0.3:  # 30%概率应用增强
            sample = self._apply_augmentation(sample)
            
        # 其余处理与原始SFTDataset相同
        prompt = self._create_chat_prompt(sample['conversations'])
        input_ids = self.tokenizer(prompt).input_ids[:self.max_length]
        input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))
        
        loss_mask = self._generate_loss_mask(input_ids)
        
        X = torch.tensor(input_ids[:-1], dtype=torch.long)
        Y = torch.tensor(input_ids[1:], dtype=torch.long)
        loss_mask = torch.tensor(loss_mask[1:], dtype=torch.long)
        
        return X, Y, loss_mask
    
    def _apply_augmentation(self, sample):
        """应用简单的数据增强技术"""
        augmented_sample = {'conversations': []}
        
        for turn in sample['conversations']:
            content = turn['content']
            
            # 随机选择增强方法
            aug_type = random.choice(['none', 'synonym', 'word_dropout'])
            
            if aug_type == 'synonym':
                # 简单的同义词替换（实际应用中可使用更复杂的同义词库）
                words = content.split()
                if len(words) > 5:  # 只对较长的句子应用
                    idx = random.randint(0, len(words) - 1)
                    # 这里仅作示例，实际应用中应使用同义词库
                    synonyms = {
                        '你好': ['您好', '嗨', '你好啊'],
                        '谢谢': ['感谢', '多谢', '谢谢你'],
                        '问题': ['疑问', '难题', '困惑'],
                        # 更多同义词...
                    }
                    for word, replacements in synonyms.items():
                        if words[idx] == word:
                            words[idx] = random.choice(replacements)
                            break
                    content = ' '.join(words)
            
            elif aug_type == 'word_dropout':
                # 随机丢弃词语（仅对用户输入应用）
                if turn['role'] == 'user':
                    words = content.split()
                    if len(words) > 5:  # 只对较长的句子应用
                        dropout_idx = random.randint(0, len(words) - 1)
                        words.pop(dropout_idx)
                        content = ' '.join(words)
            
            augmented_sample['conversations'].append({
                'role': turn['role'],
                'content': content
            })
        
        return augmented_sample
```

### 10.5 添加早停机制

可以添加早停机制，在验证损失不再改善时停止训练：

```python
class EarlyStopping:
    """早停机制，在验证损失不再改善时停止训练"""
    def __init__(self, patience=3, min_delta=0.0001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False
        
    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                
        return self.early_stop
```

在主训练循环中使用早停机制：

```python
early_stopping = EarlyStopping(patience=args.early_stopping_patience)

for epoch in range(args.epochs):
    train_epoch(epoch, wandb)
    
    # 在每个轮次结束后评估验证损失
    if args.use_validation:
        val_loss = evaluate(model, val_loader, args.device)
        Logger(f'Epoch {epoch+1} validation loss: {val_loss:.4f}')
        
        if wandb is not None and (not ddp or dist.get_rank() == 0):
            wandb.log({"val_loss": val_loss, "epoch": epoch})
        
        # 检查是否应该早停
        if early_stopping(val_loss):
            Logger(f'Early stopping triggered after epoch {epoch+1}')
            break
```

## 11. 开发者信息

本脚本是MiniMind项目的一部分，用于对预训练模型进行监督微调，使模型具备对话能力。监督微调是构建对话语言模型的关键步骤，通过高质量的指令-回复对训练模型学习遵循人类指令的能力。