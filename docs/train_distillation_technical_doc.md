# MiniMind 知识蒸馏训练脚本技术文档

## 1. 概述

`train_distillation.py` 是 MiniMind 项目中用于实现知识蒸馏（Knowledge Distillation）的训练脚本。该脚本通过将大型教师模型（Teacher Model）的知识迁移到小型学生模型（Student Model）中，实现模型压缩和性能优化，在保持模型能力的同时显著减少参数量和计算资源需求。

## 2. 主要功能

- 支持从大型模型向小型模型进行知识蒸馏
- 实现混合损失函数，结合交叉熵损失和蒸馏损失
- 支持可调节的温度参数，控制软标签的平滑程度
- 支持可调节的损失权重系数，平衡真实标签和软标签的重要性
- 支持分布式训练，实现单机多卡训练
- 支持梯度累积，有效扩大批次大小
- 实现混合精度训练，提高训练效率
- 支持余弦学习率调度，优化训练过程
- 提供 Weights & Biases (wandb) 集成，方便训练监控
- 定期保存蒸馏模型权重，确保训练成果不丢失

## 3. 知识蒸馏技术原理

知识蒸馏是一种模型压缩技术，其核心思想是：

1. 使用预训练的大型模型（教师模型）生成软标签（soft labels）
2. 训练小型模型（学生模型）同时学习真实标签和教师模型生成的软标签
3. 通过温度参数调整软标签的平滑程度，使学生模型能够学习到教师模型的决策边界和类别相似性信息

在本实现中，知识蒸馏的核心公式如下：

```
L_total = α * L_CE(student, hard_labels) + (1-α) * L_KL(student, teacher)
```

其中：
- `L_CE` 是交叉熵损失，衡量学生模型与真实标签之间的差距
- `L_KL` 是KL散度损失，衡量学生模型与教师模型输出分布之间的差距
- `α` 是权重系数，用于平衡两种损失的重要性
- 温度参数 `T` 用于控制软标签的平滑程度，较高的温度会产生更平滑的分布

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

### 4.3 `distillation_loss_fn(student_logits, teacher_logits, temperature=1.0, reduction='batchmean')`

**功能**：计算学生模型和教师模型之间的知识蒸馏损失

**参数**：
- `student_logits`：学生模型的输出logits
- `teacher_logits`：教师模型的输出logits
- `temperature`：温度参数，控制软标签的平滑程度，默认为1.0
- `reduction`：损失归约方式，默认为'batchmean'

**返回值**：
- 计算得到的KL散度损失值

**处理流程**：
1. 使用softmax函数和温度参数将教师模型的logits转换为概率分布
2. 使用log_softmax函数和温度参数将学生模型的logits转换为对数概率
3. 计算两个分布之间的KL散度
4. 乘以温度的平方作为最终损失（理论上需要乘以温度的平方来平衡梯度）

### 4.4 `train_epoch(epoch, wandb, alpha=0.0, temperature=1.0)`

**功能**：执行一个训练轮次

**参数**：
- `epoch`：当前训练轮次
- `wandb`：Weights & Biases 对象，用于记录训练指标
- `alpha`：损失权重系数，控制交叉熵损失和蒸馏损失的比例，默认为0.0
- `temperature`：温度参数，控制软标签的平滑程度，默认为1.0

**处理流程**：
1. 将教师模型设置为评估模式并冻结梯度
2. 遍历数据加载器中的每个批次
3. 将数据移至指定设备
4. 根据当前步骤计算学习率并更新优化器
5. 使用混合精度上下文执行学生模型的前向传播
6. 使用无梯度上下文执行教师模型的前向传播
7. 计算交叉熵损失和蒸馏损失
8. 根据权重系数组合两种损失
9. 执行反向传播
10. 根据梯度累积步数执行优化器步骤
11. 定期记录和打印训练状态
12. 定期保存学生模型权重

### 4.5 `init_student_model(lm_config)`

**功能**：初始化并加载学生模型和分词器

**参数**：
- `lm_config`：学生模型的配置对象

**返回值**：
- 初始化好的学生模型
- 对应的分词器

**处理流程**：
1. 加载分词器
2. 创建学生模型实例
3. 加载预训练或SFT模型权重
4. 打印学生模型参数量
5. 将模型移至指定设备
6. 返回模型和分词器

### 4.6 `init_teacher_model(lm_config)`

**功能**：初始化并加载教师模型

**参数**：
- `lm_config`：教师模型的配置对象

**返回值**：
- 初始化好的教师模型

**处理流程**：
1. 创建教师模型实例
2. 加载预训练或SFT模型权重
3. 打印教师模型参数量
4. 将模型移至指定设备
5. 返回模型

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
| `--epochs` | 6 | int | 训练轮数 |
| `--batch_size` | 32 | int | 每个批次的样本数 |
| `--learning_rate` | 5e-6 | float | 基础学习率 |
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
| `--data_path` | './dataset/sft_data.jsonl' | str | 训练数据路径 |

## 6. 数据格式说明

知识蒸馏使用与 SFT 相同的数据格式，由 `SFTDataset` 类处理。训练数据应为 JSONL 格式，每行包含一个 JSON 对象，格式如下：

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
# 使用默认配置进行知识蒸馏训练
python train_distillation.py
```

### 7.2 自定义学生和教师模型

在脚本中修改模型配置：

```python
# 定义学生模型和教师模型
lm_config_student = LMConfig(dim=512, n_layers=8, max_seq_len=512)
lm_config_teacher = LMConfig(dim=768, n_layers=16, max_seq_len=512)
```

### 7.3 使用分布式训练

```bash
# 使用torchrun启动分布式训练（2个GPU）
torchrun --nproc_per_node 2 train_distillation.py --ddp
```

### 7.4 自定义训练配置

```bash
# 自定义批次大小、学习率和训练轮数
python train_distillation.py --batch_size 64 --learning_rate 1e-5 --epochs 10
```

### 7.5 启用Weights & Biases监控

```bash
# 启用wandb监控训练过程
python train_distillation.py --use_wandb --wandb_project "MiniMind-Distillation"
```

## 8. 注意事项

1. 知识蒸馏训练前需确保教师模型和学生模型的权重文件存在于 `./out/` 目录中，默认加载的是 SFT 模型权重。

2. 温度参数（temperature）对蒸馏效果有显著影响，较高的温度会产生更平滑的软标签分布，有助于学生模型学习类别之间的相似性，但温度过高可能导致信息过度平滑。建议在1.0-5.0之间调整。

3. 损失权重系数（alpha）用于平衡交叉熵损失和蒸馏损失的重要性，alpha=0表示仅使用蒸馏损失，alpha=1表示仅使用交叉熵损失。建议在0.1-0.5之间调整，以保持对教师知识的学习同时不忽视真实标签。

4. 学生模型的学习率通常应小于普通训练，默认为5e-6，这有助于学生模型稳定地学习教师模型的知识。

5. 蒸馏模型权重会保存在 `{args.save_dir}/full_dist_{lm_config_student.dim}{moe_path}.pth` 路径下，其中 `moe_path` 表示是否使用了混合专家模型。

6. 对于大规模训练，建议启用 Weights & Biases 监控，可以通过 `--use_wandb` 参数开启。

7. 知识蒸馏训练的批次大小通常可以设置得比普通训练大，因为教师模型已经学习了良好的表示，学生模型可以更有效地从中学习。

8. 在训练过程中，可以通过观察交叉熵损失和蒸馏损失的变化来调整 alpha 参数，以获得更好的平衡。

## 9. 流程图

```
初始化
  ↓
解析命令行参数
  ↓
创建学生和教师模型配置
  ↓
初始化分布式环境（如果启用）
  ↓
初始化Weights & Biases（如果启用）
  ↓
加载学生模型和分词器
  ↓
加载教师模型
  ↓
创建数据集和数据加载器
  ↓
初始化优化器和梯度缩放器
  ↓
循环训练每个轮次
  ↓
  将教师模型设置为评估模式
  ↓
  计算学习率
  ↓
  学生模型前向传播
  ↓
  教师模型前向传播
  ↓
  计算交叉熵损失和蒸馏损失
  ↓
  组合损失并反向传播
  ↓
  梯度累积和裁剪
  ↓
  优化器步骤
  ↓
  记录训练状态
  ↓
  定期保存学生模型权重
```

## 10. 可扩展方法

### 10.1 添加验证集评估

可以扩展脚本以在训练过程中评估学生模型在验证集上的性能：

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

### 10.2 实现动态温度调整

可以实现动态温度调整策略，随着训练的进行逐渐降低温度：

```python
def get_temperature(current_step, total_steps, initial_temp=5.0, final_temp=1.0):
    """随着训练进行逐渐降低温度"""
    decay = (initial_temp - final_temp) * (current_step / total_steps)
    return initial_temp - decay
```

### 10.3 添加特征蒸馏

除了输出层的蒸馏外，还可以添加中间层特征的蒸馏，帮助学生模型更好地学习教师模型的内部表示：

```python
def feature_distillation_loss(student_features, teacher_features):
    """计算特征蒸馏损失"""
    loss = 0
    for sf, tf in zip(student_features, teacher_features):
        # 确保特征维度匹配
        if sf.size() != tf.size():
            # 可以使用线性投影或平均池化等方法调整维度
            sf = nn.functional.adaptive_avg_pool2d(sf, tf.size()[2:])
        loss += nn.functional.mse_loss(sf, tf)
    return loss / len(student_features)
```

### 10.4 实现渐进式蒸馏

可以实现渐进式蒸馏，先训练一个中等大小的模型，然后再将知识蒸馏到更小的模型：

```python
def progressive_distillation(large_model, medium_model, small_model, train_loader, epochs=5):
    """渐进式蒸馏"""
    # 第一阶段：大模型到中等模型的蒸馏
    distill(teacher=large_model, student=medium_model, train_loader=train_loader, epochs=epochs)
    
    # 第二阶段：中等模型到小模型的蒸馏
    distill(teacher=medium_model, student=small_model, train_loader=train_loader, epochs=epochs)
    
    return small_model
```

## 11. 开发者信息

本脚本是MiniMind项目的一部分，用于实现模型压缩和知识迁移。知识蒸馏技术通过让小模型学习大模型的"暗知识"，在保持模型性能的同时显著减少了参数量和计算资源需求。这使得在资源受限的设备上部署大型语言模型成为可能，为模型的实际应用提供了更灵活的解决方案。full_dist_{lm_config_