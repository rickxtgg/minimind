# MiniMind 模型评估脚本技术文档

## 1. 概述

`eval_model.py` 是 MiniMind 项目中用于评估和测试语言模型的脚本。该脚本支持多种模型配置和评估模式，包括预训练模型、SFT模型、RLHF模型和推理模型的测试，以及支持LoRA微调模型的加载和评估。

## 2. 主要功能

- 加载不同类型和配置的 MiniMind 模型
- 支持原生 PyTorch 权重和 Transformers 格式模型的加载
- 提供预设的测试问题集，针对不同模型类型和领域
- 支持流式输出（stream）和批量输出模式
- 支持历史对话上下文的管理
- 提供随机种子设置，确保结果可复现

## 3. 主要函数说明

### 3.1 `init_model(args)`

**功能**：初始化并加载模型和分词器

**参数**：
- `args`：命令行参数对象，包含模型配置信息

**返回值**：
- 初始化好的模型（设置为评估模式并移至指定设备）
- 对应的分词器

**处理流程**：
1. 加载分词器
2. 根据 `args.load` 参数决定加载方式：
   - 当 `args.load=0` 时，使用原生 PyTorch 权重加载
     - 根据模型模式（预训练、SFT、RLHF、推理）选择对应的权重文件
     - 创建 MiniMindLM 模型实例并加载权重
     - 如果指定了 LoRA 名称，则应用 LoRA 并加载对应权重
   - 当 `args.load=1` 时，使用 Transformers 格式加载模型
3. 打印模型参数量
4. 返回设置为评估模式的模型和分词器

### 3.2 `get_prompt_datas(args)`

**功能**：根据模型模式和 LoRA 配置获取测试提示数据

**参数**：
- `args`：命令行参数对象，包含模型模式和 LoRA 配置

**返回值**：
- 测试提示数据列表

**处理流程**：
1. 如果是预训练模型（`args.model_mode=0`），返回预训练模型的测试提示（接龙能力测试）
2. 如果是其他模型：
   - 当未使用 LoRA 时，返回通用对话问题
   - 当使用 LoRA 时，根据 LoRA 名称返回特定领域的问题（如身份识别、医疗等）

### 3.3 `setup_seed(seed)`

**功能**：设置随机种子，确保结果可复现

**参数**：
- `seed`：随机种子值

**处理流程**：
1. 设置 Python 的 `random` 模块种子
2. 设置 NumPy 的随机种子
3. 设置 PyTorch 的 CPU 和 CUDA 随机种子
4. 设置 CUDNN 的确定性和基准选项

### 3.4 `main()`

**功能**：脚本的主函数，处理命令行参数并执行模型评估

**处理流程**：
1. 解析命令行参数
2. 初始化模型和分词器
3. 获取测试提示数据
4. 询问用户选择测试模式（自动测试或手动输入）
5. 对每个提示：
   - 设置随机种子
   - 管理对话历史
   - 应用聊天模板（对于非预训练模型）
   - 使用模型生成回答
   - 处理流式或非流式输出
   - 更新对话历史

## 4. 命令行参数说明

| 参数 | 默认值 | 类型 | 说明 |
|------|--------|------|------|
| `--lora_name` | 'None' | str | LoRA 模型名称，用于加载特定领域的 LoRA 权重 |
| `--out_dir` | 'out' | str | 输出目录，存放模型权重文件的位置 |
| `--temperature` | 0.85 | float | 生成温度，控制输出的随机性 |
| `--top_p` | 0.85 | float | 核采样参数，控制词汇分布的截断 |
| `--device` | 'cuda'/'cpu' | str | 运行设备，自动检测 CUDA 可用性 |
| `--dim` | 512 | int | 模型隐藏层维度 |
| `--n_layers` | 8 | int | 模型层数 |
| `--max_seq_len` | 8192 | int | 最大序列长度 |
| `--use_moe` | False | bool | 是否使用混合专家模型 (MoE) |
| `--history_cnt` | 0 | int | 携带的历史对话上下文条数（偶数） |
| `--stream` | True | bool | 是否使用流式输出 |
| `--load` | 0 | int | 加载方式：0为原生torch权重，1为transformers格式 |
| `--model_mode` | 1 | int | 模型模式：0为预训练，1为SFT，2为RLHF，3为Reason |

## 5. 模型配置说明

脚本支持以下预设的模型配置：

- **MiniMind2-moe (145M)**：dim=640, n_layers=8, use_moe=True
- **MiniMind2-Small (26M)**：dim=512, n_layers=8
- **MiniMind2 (104M)**：dim=768, n_layers=16

## 6. 使用示例

### 6.1 基本使用

```bash
# 使用默认配置（SFT模型，Small版本）评估模型
python eval_model.py
```

### 6.2 加载RLHF模型

```bash
# 加载RLHF模型进行评估
python eval_model.py --model_mode 2
```

### 6.3 使用LoRA微调模型

```bash
# 加载医疗领域的LoRA微调模型
python eval_model.py --lora_name lora_medical
```

### 6.4 使用MoE模型

```bash
# 加载MoE版本的模型
python eval_model.py --dim 640 --use_moe True
```

### 6.5 使用Transformers格式模型

```bash
# 使用Transformers格式加载模型
python eval_model.py --load 1
```

## 7. 注意事项

1. `max_seq_len` 参数设置的是最大允许输入长度，并不意味着模型具有对应的长文本处理性能，主要是为了防止QA出现被截断的问题。

2. `history_cnt` 需要设置为偶数，因为每组历史记录包含一个用户问题和一个模型回答。设置为0时表示不携带历史上下文。

3. 模型未经过外推微调时，在更长的上下文中使用chat_template可能会出现性能明显退化，因此需要注意`history_cnt`的设置。

4. 使用`stream=True`可以实现流式输出，提供更好的用户体验，特别是对于长回答。

5. 可以通过设置固定的随机种子（如使用`setup_seed(2025)`而不是随机值）来确保每次运行得到相同的输出结果。

## 8. 流程图

```
初始化
  ↓
解析命令行参数
  ↓
加载模型和分词器 ← 加载LoRA权重（如果指定）
  ↓
获取测试提示数据
  ↓
选择测试模式（自动/手动）
  ↓
循环处理每个提示
  ↓
设置随机种子
  ↓
管理对话历史
  ↓
应用聊天模板
  ↓
模型生成回答
  ↓
输出结果（流式/非流式）
  ↓
更新对话历史
```

## 9. 可扩展方法

### 9.1 添加自定义评估指标

`eval_model.py` 脚本可以扩展以包含更多自定义评估指标，以便更全面地评估模型性能：

```python
# 在 eval_model.py 中添加评估指标函数
def evaluate_response(prompt, response, metrics=None):
    """评估模型回答的质量"""
    if metrics is None:
        metrics = ["length", "response_time"]
    
    results = {}
    
    # 基础指标
    if "length" in metrics:
        results["length"] = len(response)
    
    if "response_time" in metrics:
        # 需要在生成前后记录时间
        results["response_time"] = time_elapsed
    
    # 可以添加更多自定义指标
    if "keyword_match" in metrics:
        # 检查回答中是否包含特定关键词
        keywords = ["关键词1", "关键词2"] # 可自定义关键词列表
        results["keyword_match"] = sum(1 for kw in keywords if kw in response)
    
    return results
```

要添加新的评估指标，可以按照以下步骤操作：

1. 在 `evaluate_response` 函数中定义新的指标计算方法
2. 在 `main()` 函数中的模型生成部分调用评估函数
3. 收集并展示评估结果

常见的可添加指标包括：
- **响应时间**：测量模型生成回答所需的时间
- **回答长度**：统计生成回答的字符数或token数
- **关键词匹配**：检查回答中是否包含预期的关键词
- **主题一致性**：评估回答是否与问题主题相关
- **语法正确性**：使用外部工具评估语法正确性

### 9.2 扩展支持新的模型架构

要扩展脚本以支持新的模型架构或配置，可以修改 `init_model()` 函数：

```python
def init_model(args):
    tokenizer = AutoTokenizer.from_pretrained('./model/minimind_tokenizer')
    
    # 添加新的模型架构支持
    if args.model_arch == "transformer":
        # 现有的MiniMindLM模型加载逻辑
        if args.load == 0:
            # 原有代码...
        else:
            # 原有代码...
    elif args.model_arch == "new_arch":
        # 新架构模型的加载逻辑
        from model.new_model import NewArchModel
        model = NewArchModel(config)
        # 加载权重...
    elif args.model_arch == "external":
        # 支持外部模型
        model = AutoModelForCausalLM.from_pretrained(args.model_path, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    
    return model.eval().to(args.device), tokenizer
```

同时，需要在 `main()` 函数中添加相应的命令行参数：

```python
parser.add_argument('--model_arch', default='transformer', type=str, 
                    help="模型架构: transformer, new_arch, external")
parser.add_argument('--model_path', default=None, type=str, 
                    help="外部模型路径，当model_arch=external时使用")
```

### 9.3 自定义提示数据集

可以扩展 `get_prompt_datas()` 函数以支持自定义提示数据集：

```python
def get_prompt_datas(args):
    # 现有代码...
    
    # 添加从文件加载提示数据的功能
    if args.prompt_file:
        try:
            with open(args.prompt_file, 'r', encoding='utf-8') as f:
                if args.prompt_file.endswith('.json'):
                    import json
                    custom_prompts = json.load(f)
                else:
                    # 假设每行一个提示
                    custom_prompts = [line.strip() for line in f if line.strip()]
                return custom_prompts
        except Exception as e:
            print(f"加载自定义提示文件失败: {e}")
            # 返回默认提示数据
    
    return prompt_datas
```

在 `main()` 函数中添加相应的命令行参数：

```python
parser.add_argument('--prompt_file', default=None, type=str, 
                    help="自定义提示数据文件路径，支持txt(每行一个提示)或json格式")
```

自定义提示数据文件格式示例：

**JSON格式**：
```json
[
  "这是第一个提示问题？",
  "这是第二个提示问题？",
  "这是第三个提示问题？"
]
```

**TXT格式**（每行一个提示）：
```
这是第一个提示问题？
这是第二个提示问题？
这是第三个提示问题？
```

### 9.4 高级参数调优指南

为了获得最佳的模型生成效果，可以调整以下参数：

#### 9.4.1 温度（Temperature）调优

温度参数控制生成文本的随机性：
- 较低的温度（如0.2-0.5）：生成更确定性、更保守的回答
- 中等温度（如0.7-0.8）：平衡创造性和一致性
- 较高的温度（如0.9-1.2）：生成更多样化、更创造性的回答

可以根据不同任务类型调整温度：
- 事实性问答：使用较低温度（0.2-0.5）
- 创意写作：使用较高温度（0.8-1.2）
- 一般对话：使用中等温度（0.7-0.8）

#### 9.4.2 Top-p（核采样）调优

Top-p参数控制词汇分布的截断：
- 较低的top-p值（如0.5）：生成更保守、更可预测的文本
- 较高的top-p值（如0.9）：允许更多样化的词汇选择

建议的调优方法：
- 将top-p与temperature结合使用，例如(temperature=0.8, top_p=0.9)
- 对于需要创造性的任务，可以提高top-p值
- 对于需要准确性的任务，可以降低top-p值

#### 9.4.3 上下文长度调优

`history_cnt`参数控制携带的历史对话上下文条数：
- 增加上下文长度可以提高模型对话的连贯性
- 但过长的上下文可能导致性能下降，特别是对于未经过长文本微调的模型

建议根据模型的训练情况和实际需求调整上下文长度：
- 对于简单问答：设置为0或2
- 对于多轮对话：设置为4-6（即2-3轮对话历史）
- 对于复杂任务：可以尝试更长的上下文，但需要监控性能变化

可以添加动态上下文管理功能：

```python
def manage_context(messages, max_tokens, tokenizer):
    """动态管理上下文长度，确保不超过最大token限制"""
    context = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    tokens = tokenizer(context)['input_ids']
    
    # 如果超出最大长度，逐步减少历史消息
    while len(tokens) > max_tokens and len(messages) > 1:
        # 移除最早的一组对话（用户问题和模型回答）
        messages = messages[2:]
        context = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        tokens = tokenizer(context)['input_ids']
    
    return messages
```

### 9.5 与其他评估工具的集成

`eval_model.py` 脚本可以与其他评估工具集成，以提供更全面的模型评估：

#### 9.5.1 与ROUGE、BLEU等指标集成

```python
# 安装依赖: pip install rouge-score nltk
from rouge_score import rouge_scorer
import nltk
from nltk.translate.bleu_score import sentence_bleu

def calculate_metrics(reference, hypothesis):
    """计算ROUGE和BLEU分数"""
    # 初始化ROUGE计算器
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    # 计算ROUGE分数
    rouge_scores = scorer.score(reference, hypothesis)
    
    # 计算BLEU分数
    reference_tokens = nltk.word_tokenize(reference)
    hypothesis_tokens = nltk.word_tokenize(hypothesis)
    bleu_score = sentence_bleu([reference_tokens], hypothesis_tokens)
    
    return {
        'rouge1': rouge_scores['rouge1'].fmeasure,
        'rouge2': rouge_scores['rouge2'].fmeasure,
        'rougeL': rouge_scores['rougeL'].fmeasure,
        'bleu': bleu_score
    }
```

#### 9.5.2 与外部评估服务集成

```python
import requests

def evaluate_with_external_service(prompt, response, service_url):
    """使用外部评估服务评估模型回答"""
    try:
        data = {
            "prompt": prompt,
            "response": response
        }
        result = requests.post(service_url, json=data)
        return result.json()
    except Exception as e:
        print(f"外部评估服务调用失败: {e}")
        return {}
```

#### 9.5.3 批量评估与结果导出

```python
import pandas as pd
import json

def batch_evaluate(model, tokenizer, prompts, args):
    """批量评估模型并导出结果"""
    results = []
    
    for prompt in prompts:
        # 生成回答...
        # 计算评估指标...
        
        results.append({
            "prompt": prompt,
            "response": response,
            "metrics": metrics
        })
    
    # 导出结果
    if args.output_format == 'csv':
        pd.DataFrame(results).to_csv(args.output_file, index=False)
    elif args.output_format == 'json':
        with open(args.output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
    
    return results
```

在 `main()` 函数中添加相应的命令行参数：

```python
parser.add_argument('--batch_eval', action='store_true', help="启用批量评估模式")
parser.add_argument('--output_file', default='eval_results.csv', type=str, help="评估结果输出文件")
parser.add_argument('--output_format', default='csv', type=str, choices=['csv', 'json'], help="输出格式")
```

## 10. 开发者信息

本脚本是MiniMind项目的一部分，用于评估和测试不同配置和训练阶段的MiniMind语言模型。