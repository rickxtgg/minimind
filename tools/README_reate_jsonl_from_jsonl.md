# JSONL文件处理工具使用说明

## 简介

`create_jsonl_from_jsonl.py`是一个功能强大的JSONL文件处理工具，用于从输入的JSONL文件中提取、过滤、转换数据并保存到新的JSONL文件。该工具支持多种处理方式，包括顺序处理和随机抽样，并提供了性能优化选项和断点续传功能，适合处理大型JSONL数据集。

## 主要功能

1. **数据提取**：支持按行数提取或随机抽样方式提取数据
2. **数据过滤**：支持基于键值的过滤条件（普通文本匹配或正则表达式匹配）
3. **数据转换**：支持移除指定的键和自定义转换函数
4. **数据验证**：支持JSON数据基本验证和JSON Schema验证
5. **性能优化**：支持多线程处理以提高性能
6. **断点续传**：支持断点续传，避免长时间处理中断后需要重新开始
7. **运行模式**：提供交互式模式和命令行参数模式两种运行方式

## 安装依赖

使用前请确保已安装以下依赖：

```bash
pip install tqdm
```

如果需要使用JSON Schema验证功能，还需安装：

```bash
pip install jsonschema
```

## 使用方法

### 1. 交互式模式

交互式模式通过提示引导用户输入必要参数：

```bash
python create_jsonl_from_jsonl.py
```

或

```bash
python create_jsonl_from_jsonl.py -m 0
```

在交互式模式下，程序会依次提示输入：
- 输入文件路径
- 输出文件路径
- 处理行数
- 是否随机抽样
- 是否使用多线程处理
- 最大线程数（如果使用多线程）
- 批处理大小
- 断点续传文件路径（可选）
- 是否验证JSON数据
- 日志文件路径（可选）
- 要移除的键列表（可选）
- 过滤条件（可选）

### 2. 命令行参数模式

命令行参数模式允许通过命令行直接指定所有参数：

```bash
python create_jsonl_from_jsonl.py -m 1 -i input.jsonl -o output.jsonl [其他参数]
```

## 命令行参数详解

### 基本参数

| 参数 | 长参数 | 描述 | 默认值 | 示例值 |
|------|--------|------|--------|--------|
| `-i` | `--input` | 输入的JSONL文件路径 | 必填 | `data.jsonl` |
| `-o` | `--output` | 输出的JSONL文件路径 | 必填 | `output.jsonl` |
| `-n` | `--num-lines` | 要处理的行数 | 100000 | `5000` |
| `-r` | `--random` | 是否随机抽样 | False | `True` |
| `-m` | `--mode` | 运行模式: 0为交互模式, 1为命令行参数模式 | 0 | `1` |

### 性能优化参数

| 参数 | 长参数 | 描述 | 默认值 | 示例值 |
|------|--------|------|--------|--------|
| `-t` | `--threads` | 是否使用多线程处理 | False | `True` |
| `-w` | `--workers` | 多线程处理时的最大线程数 | 4 | `8` |
| `-b` | `--batch-size` | 批处理大小 | 1000 | `500` |

### 断点续传和日志参数

| 参数 | 长参数 | 描述 | 默认值 | 示例值 |
|------|--------|------|--------|--------|
| `-c` | `--checkpoint` | 断点续传文件路径 | 无 | `checkpoint.pkl` |
| `-l` | `--log-file` | 日志文件路径 | 无 | `process.log` |

### 数据验证参数

| 参数 | 长参数 | 描述 | 默认值 | 示例值 |
|------|--------|------|--------|--------|
| `-v` | `--validate` | 是否验证JSON数据 | False | `True` |

### 过滤条件参数

| 参数 | 长参数 | 描述 | 默认值 | 示例值 |
|------|--------|------|--------|--------|
| 无 | `--filter-key` | 要过滤的键名 | 无 | `type` |
| 无 | `--filter-pattern` | 过滤的模式 | 无 | `article` |
| 无 | `--filter-regex` | 是否使用正则表达式 | False | `True` |

### 数据转换参数

| 参数 | 长参数 | 描述 | 默认值 | 示例值 |
|------|--------|------|--------|--------|
| 无 | `--remove-keys` | 要移除的键列表 | 无 | `id timestamp` |

## 使用示例

### 示例1：从input.jsonl中提取10000行数据到output.jsonl

```bash
python create_jsonl_from_jsonl.py -m 1 -i input.jsonl -o output.jsonl -n 10000
```

### 示例2：随机抽样5000行数据

```bash
python create_jsonl_from_jsonl.py -m 1 -i input.jsonl -o output.jsonl -n 5000 -r
```

### 示例3：使用多线程处理并设置断点续传

```bash
python create_jsonl_from_jsonl.py -m 1 -i input.jsonl -o output.jsonl -t -w 8 -c checkpoint.pkl
```

### 示例4：过滤包含特定值的数据

```bash
python create_jsonl_from_jsonl.py -m 1 -i input.jsonl -o output.jsonl --filter-key "type" --filter-pattern "article"
```

### 示例5：移除特定键

```bash
python create_jsonl_from_jsonl.py -m 1 -i input.jsonl -o output.jsonl --remove-keys id timestamp
```

### 示例6：使用正则表达式过滤

```bash
python create_jsonl_from_jsonl.py -m 1 -i input.jsonl -o output.jsonl --filter-key "content" --filter-pattern "^[A-Z].*" --filter-regex True
```

### 示例7：启用JSON验证并记录日志

```bash
python create_jsonl_from_jsonl.py -m 1 -i input.jsonl -o output.jsonl -v -l process.log
```

## 交互参数详解示例

以下是在交互模式下运行时的参数输入示例：

```
$ python create_jsonl_from_jsonl.py

请输入输入文件路径: data.jsonl
请输入输出文件路径: filtered_data.jsonl
请输入要处理的行数 [100000]: 5000
是否随机抽样? (y/n) [n]: y
是否使用多线程处理? (y/n) [n]: y
请输入最大线程数 [4]: 8
请输入批处理大小 [1000]: 500
请输入断点续传文件路径 (可选): checkpoint.pkl
是否验证JSON数据? (y/n) [n]: n
请输入日志文件路径 (可选): process.log
请输入要移除的键列表 (用空格分隔，可选): id timestamp
是否添加过滤条件? (y/n) [n]: y
请输入要过滤的键名: type
请输入过滤的模式: article
是否使用正则表达式? (y/n) [n]: n
是否添加更多过滤条件? (y/n) [n]: y
请输入要过滤的键名: status
请输入过滤的模式: published
是否使用正则表达式? (y/n) [n]: n
是否添加更多过滤条件? (y/n) [n]: n

开始处理...
100%|██████████| 5000/5000 [00:12<00:00, 401.23it/s]
处理完成! 共处理5000行，输出3254行
```

在交互模式下，程序会提示用户输入各种参数，并在方括号中显示默认值。用户可以直接按回车键使用默认值，或输入新的值。对于是/否类型的问题，用户可以输入`y`或`n`。

对于可选参数，用户可以直接按回车键跳过。对于可以添加多个的参数（如过滤条件），程序会询问是否添加更多，直到用户输入`n`为止。


## 高级功能

### 1. 批处理大小调整

批处理大小影响内存使用，对于内存有限的环境可以减小该值：

```bash
python create_jsonl_from_jsonl.py -m 1 -i input.jsonl -o output.jsonl -b 500
```

### 2. 多条件过滤

可以指定多个过滤条件，所有条件都满足才会保留数据：

```bash
python create_jsonl_from_jsonl.py -m 1 -i input.jsonl -o output.jsonl --filter-key "type" --filter-pattern "article" --filter-key "status" --filter-pattern "published"
```

### 3. 断点续传

处理大文件时建议使用断点续传功能，以防处理过程中断：

```bash
python create_jsonl_from_jsonl.py -m 1 -i input.jsonl -o output.jsonl -c checkpoint.pkl
```

如果处理过程中断，再次运行相同命令会从断点处继续处理。

## 性能优化建议

1. **使用多线程**：对于CPU密集型操作，使用多线程可以显著提高处理速度
2. **调整批处理大小**：根据可用内存调整批处理大小
3. **使用断点续传**：处理大文件时启用断点续传功能
4. **减少验证开销**：只在必要时启用JSON验证

## 注意事项

1. 处理大文件时建议使用断点续传功能，以防处理过程中断
2. 使用多线程可以提高处理速度，但会增加内存使用
3. 批处理大小影响内存使用，对于内存有限的环境可以减小该值
4. 如果需要使用JSON Schema验证，请确保已安装jsonschema库
5. 过滤条件是"与"关系，即所有条件都满足才会保留数据
6. 随机抽样模式会先计算文件总行数，对于非常大的文件可能需要较长时间

## 错误处理

工具会记录处理过程中的错误，包括：
- 文件不存在错误
- JSON解析错误
- 断点文件读取错误
- 数据验证错误

错误信息会输出到控制台和日志文件（如果指定）。

## 技术细节

### 数据过滤机制

过滤条件由三元组(key, pattern, is_regex)组成：
- key: 要过滤的键名
- pattern: 过滤的模式
- is_regex: 是否使用正则表达式

如果is_regex为True，则使用正则表达式匹配；否则使用简单的文本包含匹配。

### 数据转换机制

数据转换支持两种操作：
1. 移除指定的键
2. 使用自定义转换函数转换键值（需要在代码中定义）

### 断点续传机制

断点续传使用pickle序列化保存当前处理状态，包括：
- 当前处理行号
- 已成功处理的行数

重新启动时会从保存的状态继续处理。

## 扩展开发

如需扩展功能，可以修改以下部分：

1. **添加新的转换函数**：在transform_data函数中添加新的转换逻辑
2. **增强过滤功能**：在filter_data函数中添加新的过滤逻辑
3. **添加新的验证规则**：在validate_json函数中添加新的验证逻辑

## 贡献

欢迎提交问题报告和功能请求。如果您想贡献代码，请先讨论您想要更改的内容。