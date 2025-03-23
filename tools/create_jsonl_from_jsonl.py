#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
JSONL文件处理工具 (create_jsonl_from_jsonl.py)

这个脚本用于处理JSONL文件，支持从输入的JSONL文件中提取、过滤、转换数据并保存到新的JSONL文件。
主要功能包括：
1. 支持按行数或随机抽样方式提取数据
2. 支持基于键值的过滤条件（普通文本匹配或正则表达式匹配）
3. 支持移除指定的键
4. 支持数据转换功能
5. 支持JSON数据验证
6. 支持多线程处理以提高性能
7. 支持断点续传，避免长时间处理中断后需要重新开始
8. 提供交互式模式和命令行参数模式两种运行方式

使用方法：
    1. 交互式模式：
       python create_jsonl_from_jsonl.py
       或
       python create_jsonl_from_jsonl.py -m 0
    
    2. 命令行参数模式：
       python create_jsonl_from_jsonl.py -m 1 -i input.jsonl -o output.jsonl [其他参数]

命令行参数：
    -i, --input          输入的JSONL文件路径
    -o, --output         输出的JSONL文件路径
    -n, --num-lines      要处理的行数 (默认: 100000)
    -r, --random         是否随机抽样 (默认: False)
    -t, --threads        是否使用多线程处理 (默认: False)
    -w, --workers        多线程处理时的最大线程数 (默认: 4)
    -c, --checkpoint     断点续传文件路径
    -b, --batch-size     批处理大小 (默认: 1000)
    -v, --validate       是否验证JSON数据 (默认: False)
    -l, --log-file       日志文件路径
    -m, --mode           运行模式: 0 (默认) 为交互模式, 1 为命令行参数模式
    
    过滤条件相关参数：
    --filter-key         要过滤的键名
    --filter-pattern     过滤的模式
    --filter-regex       是否使用正则表达式
    
    数据转换相关参数：
    --remove-keys        要移除的键列表

示例：
    1. 从input.jsonl中提取10000行数据到output.jsonl：
       python create_jsonl_from_jsonl.py -m 1 -i input.jsonl -o output.jsonl -n 10000
    
    2. 随机抽样5000行数据：
       python create_jsonl_from_jsonl.py -m 1 -i input.jsonl -o output.jsonl -n 5000 -r
    
    3. 使用多线程处理并设置断点续传：
       python create_jsonl_from_jsonl.py -m 1 -i input.jsonl -o output.jsonl -t -w 8 -c checkpoint.pkl
    
    4. 过滤包含特定值的数据：
       python create_jsonl_from_jsonl.py -m 1 -i input.jsonl -o output.jsonl --filter-key "type" --filter-pattern "article"
    
    5. 移除特定键：
       python create_jsonl_from_jsonl.py -m 1 -i input.jsonl -o output.jsonl --remove-keys id timestamp

注意事项：
    1. 处理大文件时建议使用断点续传功能，以防处理过程中断
    2. 使用多线程可以提高处理速度，但会增加内存使用
    3. 批处理大小影响内存使用，对于内存有限的环境可以减小该值
    4. 如果需要使用JSON Schema验证，请确保已安装jsonschema库
"""

import json
import argparse
import os
import random
import re
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import pickle
import time
import logging
import sys

# 配置日志
def setup_logging(log_file=None, log_level=logging.INFO):
    """设置日志配置"""
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=handlers
    )
    return logging.getLogger()

def transform_data(data, transform_keys=None, remove_keys=None):
    """
    转换JSON数据
    
    Args:
        data: 原始JSON数据
        transform_keys: 需要转换的键值对 {key: transform_function}
        remove_keys: 需要移除的键列表
    
    Returns:
        转换后的数据
    """
    if remove_keys:
        for key in remove_keys:
            if key in data:
                del data[key]
    
    if transform_keys:
        for key, transform_func in transform_keys.items():
            if key in data:
                data[key] = transform_func(data[key])
    
    return data

def filter_data(data, filter_conditions):
    """
    根据条件过滤数据
    
    Args:
        data: JSON数据
        filter_conditions: 过滤条件列表，每个条件是一个(key, pattern, is_regex)元组
    
    Returns:
        True表示保留，False表示过滤掉
    """
    if not filter_conditions:
        return True
        
    for key, pattern, is_regex in filter_conditions:
        if key not in data:
            return False
            
        value = str(data[key])
        if is_regex:
            if not re.search(pattern, value):
                return False
        else:
            if pattern not in value:
                return False
                
    return True

def validate_json(data, schema=None):
    """
    验证JSON数据是否符合预期格式
    
    Args:
        data: JSON数据
        schema: 可选的JSON Schema
        
    Returns:
        (bool, str): (是否有效, 错误信息)
    """
    # 基本验证 - 确保是字典类型
    if not isinstance(data, dict):
        return False, f"数据不是字典类型: {type(data)}"
    
    # 如果提供了schema，进行schema验证
    if schema:
        try:
            # 实际实现jsonschema验证
            import jsonschema
            jsonschema.validate(data, schema)
        except ImportError:
            return False, "jsonschema库未安装，无法进行schema验证"
        except jsonschema.exceptions.ValidationError as e:
            return False, f"Schema验证失败: {str(e)}"
    
    return True, ""

def process_line(line, filter_conditions=None, transform_keys=None, remove_keys=None, validate=False, schema=None):
    """处理单行JSON数据"""
    try:
        data = json.loads(line)
        
        # 应用过滤条件
        if not filter_data(data, filter_conditions):
            return None
            
        # 应用数据转换
        data = transform_data(data, transform_keys, remove_keys)
        
        # 验证数据
        if validate:
            is_valid, error = validate_json(data, schema)
            if not is_valid:
                logging.warning(f"数据验证失败 - {error}")
                return None
        
        return data
    except json.JSONDecodeError:
        return None
    except Exception as e:
        logging.error(f"处理行时出错: {str(e)}")
        return None

# 添加缺失的process_batch函数
def process_batch(batch_lines, filter_conditions=None, transform_keys=None, remove_keys=None, 
                use_threads=False, max_workers=4, validate=False, schema=None):
    """
    批量处理多行JSON数据
    
    Args:
        batch_lines: 要处理的行列表
        filter_conditions: 过滤条件
        transform_keys: 转换函数
        remove_keys: 要移除的键
        use_threads: 是否使用多线程
        max_workers: 最大线程数
        validate: 是否验证数据
        schema: JSON Schema
        
    Returns:
        处理后的数据列表
    """
    results = []
    
    if use_threads and len(batch_lines) > 1:
        # 使用多线程处理
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for line in batch_lines:
                future = executor.submit(
                    process_line, line, filter_conditions, transform_keys, remove_keys, validate, schema
                )
                futures.append(future)
            
            # 收集结果
            for future in futures:
                result = future.result()
                if result is not None:
                    results.append(result)
    else:
        # 顺序处理
        for line in batch_lines:
            result = process_line(line, filter_conditions, transform_keys, remove_keys, validate, schema)
            if result is not None:
                results.append(result)
    
    return results

def print_stats(stats):
    """打印处理统计信息"""
    print("\n" + "="*50)
    print("处理统计信息:")
    print(f"- 总处理行数: {stats['processed'] + stats['skipped']}")
    print(f"- 成功处理: {stats['processed']} 行")
    print(f"- 被过滤/跳过: {stats['skipped']} 行")
    if 'errors' in stats:
        print(f"- 错误数: {stats['errors']}")
    
    if 'duration' in stats:
        duration = stats['duration']
        if duration < 60:
            time_str = f"{duration:.2f} 秒"
        elif duration < 3600:
            time_str = f"{duration/60:.2f} 分钟"
        else:
            time_str = f"{duration/3600:.2f} 小时"
        
        print(f"- 处理时间: {time_str}")
        
        if stats['processed'] > 0 and duration > 0:
            speed = stats['processed'] / duration
            print(f"- 处理速度: {speed:.2f} 行/秒")
    
    print("="*50)

def process_jsonl(input_file, output_file, num_lines=100000, random_sample=False, 
                 filter_conditions=None, transform_keys=None, remove_keys=None,
                 use_threads=False, max_workers=4, checkpoint_file=None, batch_size=1000,
                 validate=False, schema=None, logger=None):
    """
    从指定的 JSONL 文件读取数据，并保存到一个新的 JSONL 文件。

    Args:
        input_file: 输入的 JSONL 文件路径。
        output_file: 输出的 JSONL 文件路径。
        num_lines: 要读取的行数 (默认为 100000)。
        random_sample: 是否随机抽样 (默认为 False)。
        filter_conditions: 过滤条件列表 [(key, pattern, is_regex), ...]
        transform_keys: 转换函数字典 {key: transform_function}
        remove_keys: 要移除的键列表
        use_threads: 是否使用多线程处理
        max_workers: 最大线程数
        checkpoint_file: 断点续传文件路径
        batch_size: 批处理大小，控制内存使用
        validate: 是否验证JSON数据
        schema: JSON Schema
        logger: 日志记录器
    """
    # 使用日志或打印
    log = logger.info if logger else print
    
    try:
        # 检查输入文件是否存在
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"输入文件 '{input_file}' 不存在")
            
        # 确保输出目录存在
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 检查断点续传
        start_line = 0
        processed_count = 0
        current_line = 0
        processed = 0
        skipped = 0
        
        if checkpoint_file and os.path.exists(checkpoint_file):
            try:
                with open(checkpoint_file, 'rb') as f:
                    checkpoint_data = pickle.load(f)
                    start_line = checkpoint_data.get('line', 0)
                    processed_count = checkpoint_data.get('processed', 0)
                log(f"从断点继续: 跳过前 {start_line} 行，已处理 {processed_count} 行")
            except Exception as e:
                log(f"读取断点文件失败: {e}")
        
        # 如果随机抽样，需要先计算总行数
        if random_sample:
            log("计算文件总行数...")
            total_lines = sum(1 for _ in open(input_file, 'r', encoding='utf-8'))
            log(f"文件总行数: {total_lines}")
            sample_indices = set(random.sample(range(total_lines), min(num_lines, total_lines)))
            
        # 打开输出文件，准备流式写入
        with open(output_file, 'w', encoding='utf-8') as outfile:
            # 统计数据
            stats = {
                'processed': 0,
                'skipped': 0,
                'errors': 0,
                'start_time': time.time()
            }
            
            log(f"开始处理文件: {input_file}")
            with open(input_file, 'r', encoding='utf-8') as infile:
                # 跳过已处理的行
                for _ in range(start_line):
                    next(infile, None)
                    current_line += 1
                
                # 流式处理剩余行
                if random_sample:
                    # 随机抽样模式
                    for i, line in tqdm(enumerate(infile, start=start_line), 
                                       desc="处理JSONL", unit="行"):
                        if i in sample_indices:
                            result = process_line(line, filter_conditions, transform_keys, remove_keys, validate, schema)
                            if result is not None:
                                json.dump(result, outfile, ensure_ascii=False)
                                outfile.write('\n')
                                processed += 1
                            else:
                                skipped += 1
                        current_line += 1
                        
                        # 定期更新断点
                        if checkpoint_file and current_line % 10000 == 0:
                            with open(checkpoint_file, 'wb') as f:
                                pickle.dump({
                                    'line': current_line,
                                    'processed': processed_count + processed
                                }, f)
                        
                        # 达到目标行数后退出
                        if processed >= num_lines:
                            break
                else:
                    # 顺序处理模式 - 分批处理以减少内存使用
                    batch_lines = []
                    for line in tqdm(infile, desc="读取JSONL", unit="行", total=num_lines):
                        batch_lines.append(line)
                        
                        # 当积累了一批数据或达到了目标行数时进行处理
                        if len(batch_lines) >= batch_size or processed + len(batch_lines) >= num_lines:
                            batch_results = process_batch(
                                batch_lines, 
                                filter_conditions, 
                                transform_keys, 
                                remove_keys,
                                use_threads,
                                max_workers,
                                validate,
                                schema
                            )
                            
                            # 写入结果
                            for data in batch_results:
                                json.dump(data, outfile, ensure_ascii=False)
                                outfile.write('\n')
                            
                            processed += len(batch_results)
                            skipped += len(batch_lines) - len(batch_results)
                            current_line += len(batch_lines)
                            
                            # 清空批处理列表
                            batch_lines = []
                            
                            # 达到目标行数后退出
                            if processed >= num_lines:
                                break
                            
                            # 定期更新断点
                            if checkpoint_file and current_line % 10000 == 0:
                                with open(checkpoint_file, 'wb') as f:
                                    pickle.dump({
                                        'line': start_line + current_line,
                                        'processed': processed_count + processed
                                    }, f)
            
            # 更新统计信息
            stats['processed'] = processed
            stats['skipped'] = skipped
            stats['duration'] = time.time() - stats['start_time']
            
            # 打印统计信息
            log(f"处理完成: 成功处理 {processed} 行，跳过 {skipped} 行")
            print_stats(stats)
            
            # 最终更新断点
            if checkpoint_file:
                with open(checkpoint_file, 'wb') as f:
                    pickle.dump({
                        'line': start_line + current_line,
                        'processed': processed_count + processed
                    }, f)
                log(f"断点已更新: 当前行 {start_line + current_line}, 已处理 {processed_count + processed} 行")
                
            return processed, skipped
            
    except Exception as e:
        log(f"处理过程中出错: {str(e)}")
        import traceback
        log(traceback.format_exc())
        return 0, 0


def interactive_mode():
    """交互式模式，让用户通过提示输入必要参数"""
    print("\n" + "="*50)
    print("JSONL文件处理工具 - 交互模式")
    print("="*50)
    
    # 输入文件路径
    while True:
        input_file = input("请输入JSONL输入文件路径: ").strip()
        # 去除路径前后的引号
        input_file = input_file.strip('"').strip("'")
        if os.path.exists(input_file):
            break
        else:
            print("错误: 文件不存在，请重新输入")
    
    # 输出文件路径
    output_file = input("请输入JSONL输出文件路径: ").strip()
    output_file = output_file.strip('"').strip("'")
    
    # 处理行数
    while True:
        try:
            num_lines_input = input("请输入要处理的行数 (默认: 100000): ").strip()
            num_lines = 100000 if not num_lines_input else int(num_lines_input)
            if num_lines > 0:
                break
            else:
                print("错误: 行数必须大于0")
        except ValueError:
            print("错误: 请输入有效的数字")
    
    # 是否随机抽样
    random_sample = input("是否随机抽样? (y/n, 默认: n): ").strip().lower() == 'y'
    
    # 是否使用多线程
    use_threads = input("是否使用多线程处理? (y/n, 默认: n): ").strip().lower() == 'y'
    
    # 最大线程数
    max_workers = 4
    if use_threads:
        while True:
            try:
                max_workers_input = input("请输入最大线程数 (默认: 4): ").strip()
                max_workers = 4 if not max_workers_input else int(max_workers_input)
                if max_workers > 0:
                    break
                else:
                    print("错误: 线程数必须大于0")
            except ValueError:
                print("错误: 请输入有效的数字")
    
    # 批处理大小
    while True:
        try:
            batch_size_input = input("请输入批处理大小 (默认: 1000): ").strip()
            batch_size = 1000 if not batch_size_input else int(batch_size_input)
            if batch_size > 0:
                break
            else:
                print("错误: 批处理大小必须大于0")
        except ValueError:
            print("错误: 请输入有效的数字")
    
    # 断点续传文件
    checkpoint_file = input("请输入断点续传文件路径 (可选): ").strip()
    checkpoint_file = checkpoint_file if checkpoint_file else None
    
    # 是否验证JSON数据
    validate = input("是否验证JSON数据? (y/n, 默认: n): ").strip().lower() == 'y'
    
    # 日志文件
    log_file = input("请输入日志文件路径 (可选): ").strip()
    log_file = log_file if log_file else None
    
    # 要移除的键
    remove_keys_input = input("请输入要移除的键列表 (以空格分隔, 可选): ").strip()
    remove_keys = remove_keys_input.split() if remove_keys_input else None
    
    # 过滤条件
    filter_conditions = []
    add_filter = input("是否添加过滤条件? (y/n, 默认: n): ").strip().lower() == 'y'
    
    while add_filter:
        key = input("请输入要过滤的键名: ").strip()
        pattern = input("请输入过滤的模式: ").strip()
        is_regex = input("是否使用正则表达式? (y/n, 默认: n): ").strip().lower() == 'y'
        
        filter_conditions.append((key, pattern, is_regex))
        
        add_filter = input("是否继续添加过滤条件? (y/n, 默认: n): ").strip().lower() == 'y'
    
    # 设置日志
    logger = setup_logging(log_file)
    
    # 处理JSONL文件
    try:
        processed, skipped = process_jsonl(
            input_file,
            output_file,
            num_lines=num_lines,
            random_sample=random_sample,
            filter_conditions=filter_conditions if filter_conditions else None,
            remove_keys=remove_keys,
            use_threads=use_threads,
            max_workers=max_workers,
            checkpoint_file=checkpoint_file,
            batch_size=batch_size,
            validate=validate,
            logger=logger
        )
        
        logger.info(f"处理完成: 成功处理 {processed} 行，跳过 {skipped} 行")
    except Exception as e:
        logger.error(f"处理过程中出错: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return 1
    
    return 0

def main():
    parser = argparse.ArgumentParser(description="从JSONL文件中提取和转换数据")
    parser.add_argument("-i", "--input", help="输入的JSONL文件路径")
    parser.add_argument("-o", "--output", help="输出的JSONL文件路径")
    parser.add_argument("-n", "--num-lines", type=int, default=100000, help="要处理的行数 (默认: 100000)")
    parser.add_argument("-r", "--random", action="store_true", help="是否随机抽样")
    parser.add_argument("-t", "--threads", action="store_true", help="是否使用多线程处理")
    parser.add_argument("-w", "--workers", type=int, default=4, help="多线程处理时的最大线程数 (默认: 4)")
    parser.add_argument("-c", "--checkpoint", help="断点续传文件路径")
    parser.add_argument("-b", "--batch-size", type=int, default=1000, help="批处理大小 (默认: 1000)")
    parser.add_argument("-v", "--validate", action="store_true", help="是否验证JSON数据")
    parser.add_argument("-l", "--log-file", help="日志文件路径")
    parser.add_argument("-m", "--mode", type=int, default=0, choices=[0, 1], 
                        help="运行模式: 0 (默认) 为交互模式, 1 为命令行参数模式")
    
    # 过滤条件相关参数
    filter_group = parser.add_argument_group('过滤条件')
    filter_group.add_argument("--filter-key", action="append", help="要过滤的键名")
    filter_group.add_argument("--filter-pattern", action="append", help="过滤的模式")
    filter_group.add_argument("--filter-regex", action="append", type=bool, default=False, help="是否使用正则表达式")
    
    # 转换相关参数
    transform_group = parser.add_argument_group('数据转换')
    transform_group.add_argument("--remove-keys", nargs="+", help="要移除的键列表")
    
    args = parser.parse_args()
    
    # 根据运行模式选择交互模式或命令行参数模式
    if args.mode == 0:
        # 交互模式
        return interactive_mode()
    else:
        # 命令行参数模式 - 检查必要参数
        if not args.input or not args.output:
            print("错误: 命令行参数模式需要指定输入文件(-i)和输出文件(-o)")
            parser.print_help()
            return 1
            
        # 设置日志
        logger = setup_logging(args.log_file)
        
        # 构建过滤条件
        filter_conditions = None
        if args.filter_key and args.filter_pattern:
            filter_conditions = []
            for i in range(min(len(args.filter_key), len(args.filter_pattern))):
                is_regex = args.filter_regex[i] if i < len(args.filter_regex) else False
                filter_conditions.append((args.filter_key[i], args.filter_pattern[i], is_regex))
        
        # 处理JSONL文件
        try:
            processed, skipped = process_jsonl(
                args.input,
                args.output,
                num_lines=args.num_lines,
                random_sample=args.random,
                filter_conditions=filter_conditions,
                remove_keys=args.remove_keys,
                use_threads=args.threads,
                max_workers=args.workers,
                checkpoint_file=args.checkpoint,
                batch_size=args.batch_size,
                validate=args.validate,
                logger=logger
            )
            
            logger.info(f"处理完成: 成功处理 {processed} 行，跳过 {skipped} 行")
        except Exception as e:
            logger.error(f"处理过程中出错: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return 1
        
        return 0

if __name__ == "__main__":
    sys.exit(main())
