"""
MiniMind 数据转换工具
用于将各种格式的数据转换为 MiniMind 训练所需的格式

功能说明：
1. 支持将通用 SFT 格式转换为 MiniMind 预训练格式
   - 输入格式：{"instruction": "指令", "input": "输入", "output": "输出"}
   - 输出格式：{"text": "<s>指令输入输出</s>"}
   - 适用于预训练阶段，将对话数据转换为连续文本

2. 支持将通用 SFT 格式转换为 MiniMind SFT 格式
   - 输入格式：{"instruction": "指令", "input": "输入", "output": "输出"}
   - 输出格式：{"conversations": [{"role": "user", "content": "指令\n输入"}, {"role": "assistant", "content": "输出"}]}
   - 适用于监督微调阶段，保留对话结构

3. 支持将表格数据转换为 MiniMind 预训练格式
   - 支持 CSV/Excel 格式
   - 可自定义问题列和答案列
   - 输出格式：{"text": "<s>问题答案</s>"}
   - 适用于将结构化表格数据转换为预训练文本

4. 支持将表格数据转换为 MiniMind SFT 格式
   - 支持 CSV/Excel 格式
   - 可自定义问题列和答案列
   - 输出格式：{"conversations": [{"role": "user", "content": "问题"}, {"role": "assistant", "content": "答案"}]}
   - 适用于将结构化表格数据转换为对话格式

5. 支持将 Arrow/Parquet 数据转换为 MiniMind 预训练格式
   - 支持 .arrow, .parquet 格式
   - 可自定义问题列和答案列
   - 输出格式：{"text": "<s>问题答案</s>"}
   - 适用于处理大规模数据集

6. 支持将 Arrow/Parquet 数据转换为 MiniMind SFT 格式
   - 支持 .arrow, .parquet 格式
   - 可自定义问题列和答案列
   - 输出格式：{"conversations": [{"role": "user", "content": "问题"}, {"role": "assistant", "content": "答案"}]}
   - 适用于处理大规模对话数据集

使用方法：
1. 命令行模式：
   - 转换为预训练格式：python convert_jsonl.py -m 1 -i 输入路径 [-o 输出路径]
   - 转换为SFT格式：python convert_jsonl.py -m 2 -i 输入路径 [-o 输出路径]
   - 表格转预训练：python convert_jsonl.py -m 3 -i 输入路径 [-o 输出路径] [-q 问题列] [-a 答案列]
   - 表格转SFT：python convert_jsonl.py -m 4 -i 输入路径 [-o 输出路径] [-q 问题列] [-a 答案列]
   - Arrow/Parquet转预训练：python convert_jsonl.py -m 5 -i 输入路径 [-o 输出路径] [-q 问题列] [-a 答案列]
   - Arrow/Parquet转SFT：python convert_jsonl.py -m 6 -i 输入路径 [-o 输出路径] [-q 问题列] [-a 答案列]
   - 预览数据：添加 --preview 参数可在转换前预览数据
   
2. 交互模式：
   - 直接运行：python convert_jsonl.py
   - 根据提示选择转换模式和输入路径
   - 输入 'q' 返回主菜单，输入 '0' 退出程序

参数说明：
-i, --input：输入文件路径或目录，支持单个文件或整个目录批量处理
-o, --output：输出文件路径（可选），默认在原目录生成带前缀的文件
-m, --mode：转换模式（0:交互，1:预训练，2:SFT，3:表格转预训练，4:表格转SFT，5:Arrow/Parquet转预训练，6:Arrow/Parquet转SFT）
-q, --question：问题列名或列索引（表格/Arrow/Parquet模式使用），默认使用第一列
-a, --answer：答案列名或列索引（表格/Arrow/Parquet模式使用），默认使用第二列
--preview：预览数据（显示第一行），帮助确认数据格式是否正确

示例：
1. 将 alpaca.jsonl 转换为预训练格式：
   python convert_jsonl.py -m 1 -i data/alpaca.jsonl -o data/pre_alpaca.jsonl

2. 将整个目录的 JSONL 文件转换为 SFT 格式：
   python convert_jsonl.py -m 2 -i data/sft_data/

3. 将 CSV 表格转换为预训练格式，指定问题和答案列：
   python convert_jsonl.py -m 3 -i data/qa.csv -q "问题" -a "回答"

4. 将 Parquet 文件转换为 SFT 格式并预览数据：
   python convert_jsonl.py -m 6 -i data/large_dataset.parquet --preview

注意事项：
1. 支持单个文件或整个目录的批量转换
2. 默认输出文件将添加相应前缀：预训练格式添加 "pre_"，SFT 格式添加 "sft_"
3. 转换过程会显示进度条和详细统计信息（成功/失败条数、处理速度等）
4. 表格数据支持 .csv 和 .xlsx/.xls 格式
5. 大数据格式支持 .arrow 和 .parquet 格式
6. 对于空值或无效数据会自动跳过并计入失败条数
7. 转换完成后会自动预览转换结果的第一条数据
8. 文件路径支持绝对路径和相对路径，包含空格的路径请使用引号包围
"""

import json
import os
import argparse
from tqdm import tqdm
import time
import sys
import pandas as pd  # 用于处理表格数据
import pyarrow as pa  # 用于处理Arrow格式
import pyarrow.parquet as pq  # 用于处理Parquet格式
import pyarrow.ipc  # 添加这一行以支持pa.ipc.open_file()

def convert_sft_to_minimind_sft(input_path, output_path=None):
    """将通用 SFT 数据格式的 JSONL 文件转换为 MiniMind 监督微调 (SFT) 格式。"""
    def process_sft_file(filepath, output_filepath):  # 重命名为 process_sft_file
        """处理单个 JSONL 文件为 SFT 格式。"""
        try:
            print(f"\n处理文件：{filepath}")
            print(f"输出文件：{output_filepath}")
            
            start_time = time.time()
            file_size = os.path.getsize(filepath) / (1024 * 1024)
            print(f"文件大小：{file_size:.2f} MB")

            success_count = 0
            error_count = 0
            
            with open(filepath, 'r', encoding='utf-8') as infile, \
                 open(output_filepath, 'w', encoding='utf-8') as outfile:
                total_lines = sum(1 for _ in open(filepath, 'r', encoding='utf-8'))
                print(f"总行数：{total_lines}")
                
                for line in tqdm(infile, total=total_lines, desc=f"转换 {os.path.basename(filepath)}"):
                    try:
                        data = json.loads(line)
                        instruction = data.get("instruction", "")
                        input_text = data.get("input", "")
                        output_text = data.get("output", "")

                        conversations = []
                        if instruction:
                            user_content = instruction
                            if input_text:
                                user_content += f"\n{input_text}"
                            conversations.append({"role": "user", "content": user_content})
                        if output_text:
                            conversations.append({"role": "assistant", "content": output_text})

                        new_data = {"conversations": conversations}
                        json.dump(new_data, outfile, ensure_ascii=False)
                        outfile.write('\n')
                        success_count += 1
                    
                    except json.JSONDecodeError as e:
                        error_count += 1
                        print(f"解析 JSON 行时出错（文件：{filepath}）：{e}")
                        continue

            end_time = time.time()
            duration = end_time - start_time
            
            print(f"\n处理完成:")
            print(f"- 成功转换: {success_count} 条")
            print(f"- 失败条数: {error_count} 条")
            print(f"- 处理用时: {duration:.2f} 秒")
            print(f"- 处理速度: {total_lines/duration:.2f} 条/秒")
            
            return True  # 添加返回值
            
        except FileNotFoundError:
            print(f"错误：文件未找到 - {filepath}")
        except PermissionError:
            print(f"错误：没有权限访问文件 - {filepath}")
        except Exception as e:
            print(f"错误：处理文件 {filepath} 时发生异常 - {str(e)}")
        return False  # 添加失败时的返回值

    print("\n=== 开始转换为 MiniMind SFT 格式 ===")
    start_time = time.time()
    processed_files = 0
    output_files = []  # 记录所有输出文件路径

    if os.path.isfile(input_path):
        if output_path is None:
            output_path = os.path.join(os.path.dirname(input_path), "sft_" + os.path.basename(input_path))
        if process_sft_file(input_path, output_path):  # 使用重命名后的函数
            processed_files += 1
            output_files.append(output_path)
    
    elif os.path.isdir(input_path):
        print(f"\n扫描目录：{input_path}")
        jsonl_files = [f for f in os.listdir(input_path) if f.endswith('.jsonl')]
        print(f"找到 {len(jsonl_files)} 个 JSONL 文件")
        
        for filename in jsonl_files:
            filepath = os.path.join(input_path, filename)
            # 更新命名逻辑：使用 sft_ 前缀
            output_filepath = os.path.join(input_path, "sft_" + filename)
            if process_sft_file(filepath, output_filepath):
                processed_files += 1
                output_files.append(output_filepath)  # 添加这一行
            
    duration = time.time() - start_time
    print(f"\n=== 转换完成 ===")
    print(f"- 处理文件数: {processed_files}")
    print(f"- 总用时: {duration:.2f} 秒")
    
    # 转换完成后预览第一个输出文件的第一行
    if output_files and os.path.exists(output_files[0]):
        print("\n=== 转换结果预览 ===")
        preview_data(output_files[0], 2)  # 使用模式2预览SFT格式文件

def convert_sft_to_minimind_pretrain(input_path, output_path=None):
    """将通用 SFT 数据格式的 JSONL 文件转换为 MiniMind 预训练格式。"""
    print("\n=== 开始转换为 MiniMind 预训练格式 ===")
    start_time = time.time()
    processed_files = 0  # 初始化变量
    output_files = []  # 记录所有输出文件路径

    def process_pretrain_file(filepath, output_filepath):
        """处理单个 JSONL 文件为预训练格式。"""
        try:
            print(f"\n处理文件：{filepath}")
            print(f"输出文件：{output_filepath}")
            
            start_time = time.time()
            file_size = os.path.getsize(filepath) / (1024 * 1024)
            print(f"文件大小：{file_size:.2f} MB")

            success_count = 0
            error_count = 0
            
            with open(filepath, 'r', encoding='utf-8') as infile, \
                    open(output_filepath, 'w', encoding='utf-8') as outfile:
                # 使用更高效的方式计算行数
                with open(filepath, 'r', encoding='utf-8') as count_file:
                    num_lines = sum(1 for _ in count_file)
                print(f"总行数：{num_lines}")

                for line in tqdm(infile, total=num_lines, desc=f"转换 {os.path.basename(filepath)}"):
                    try:
                        data = json.loads(line)
                        instruction = data.get("instruction", "")
                        input_text = data.get("input", "")
                        output = data.get("output", "")

                        if input_text != "":
                            text = f"<s>{instruction}{input_text}{output}</s>"
                        else:
                            text = f"<s>{instruction}{output}</s>"

                        output_data = {"text": text}
                        json.dump(output_data, outfile, ensure_ascii=False)
                        outfile.write('\n')
                        success_count += 1
                    except json.JSONDecodeError as e:
                        error_count += 1
                        print(f"解析 JSON 行时出错（文件：{filepath}）：{e}")
                        continue

            end_time = time.time()
            duration = end_time - start_time
            print(f"\n处理完成:")
            print(f"- 成功转换: {success_count} 条")
            print(f"- 失败条数: {error_count} 条")
            print(f"- 处理用时: {duration:.2f} 秒")
            print(f"- 处理速度: {num_lines/duration:.2f} 条/秒")
            
            return True  # 返回成功标志

        except FileNotFoundError:
            print(f"错误：文件未找到 - {filepath}")
        except Exception as e:
            print(f"错误：处理文件 {filepath} 时发生错误：{e}")
        return False

    if os.path.isfile(input_path):
        if output_path is None:
            # 更新命名逻辑：使用 pre_ 前缀
            base_name = os.path.basename(input_path)
            output_path = os.path.join(os.path.dirname(input_path), f"pre_{base_name}")
        if process_pretrain_file(input_path, output_path):
            processed_files += 1
            output_files.append(output_path)

    elif os.path.isdir(input_path):
        print(f"\n扫描目录：{input_path}")
        jsonl_files = [f for f in os.listdir(input_path) if f.endswith('.jsonl')]
        print(f"找到 {len(jsonl_files)} 个 JSONL 文件")
        
        for filename in jsonl_files:
            filepath = os.path.join(input_path, filename)
            # 更新命名逻辑：使用 pre_ 前缀
            output_filepath = os.path.join(input_path, f"pre_{filename}")
            if process_pretrain_file(filepath, output_filepath):
                processed_files += 1
                output_files.append(output_filepath)
    
    duration = time.time() - start_time
    print(f"\n=== 转换完成 ===")
    print(f"- 处理文件数: {processed_files}")
    print(f"- 总用时: {duration:.2f} 秒")
    
    # 转换完成后预览第一个输出文件的第一行
    if output_files and os.path.exists(output_files[0]):
        print("\n=== 转换结果预览 ===")
        preview_data(output_files[0], 1)  # 使用模式1预览预训练格式文件

def convert_table_to_minimind_pretrain(input_path, output_path=None, question_col=None, answer_col=None):
    """将表格数据转换为 MiniMind 预训练格式"""
    print("\n=== 开始转换表格数据为 MiniMind 预训练格式 ===")
    start_time = time.time()
    processed_files = 0
    output_files = []  # 记录所有输出文件路径

    def process_table_file(filepath, output_filepath, q_col, a_col):
        try:
            print(f"\n处理文件：{filepath}")
            print(f"输出文件：{output_filepath}")
            
            # 读取表格数据
            if filepath.endswith('.csv'):
                df = pd.read_csv(filepath)
            else:
                df = pd.read_excel(filepath)
            
            # 如果未指定列，使用前两列
            if q_col is None:
                q_col = df.columns[0]
            if a_col is None:
                a_col = df.columns[1]
            
            print(f"问题列：{q_col}")
            print(f"答案列：{a_col}")
            
            success_count = 0
            error_count = 0
            
            with open(output_filepath, 'w', encoding='utf-8') as outfile:
                for _, row in tqdm(df.iterrows(), total=len(df), desc="转换进度"):
                    try:
                        question = str(row[q_col]).strip()
                        answer = str(row[a_col]).strip()
                        
                        if question and answer and question.lower() != 'nan' and answer.lower() != 'nan':
                            text = f"<s>{question}{answer}</s>"
                            output_data = {"text": text}
                            json.dump(output_data, outfile, ensure_ascii=False)
                            outfile.write('\n')
                            success_count += 1
                        else:
                            error_count += 1
                            
                    except Exception as e:
                        error_count += 1
                        print(f"处理行时出错：{e}")
                        continue
            
            print(f"\n处理完成:")
            print(f"- 成功转换: {success_count} 条")
            print(f"- 失败条数: {error_count} 条")
            
        except Exception as e:
            print(f"错误：处理文件 {filepath} 时发生错误：{e}")
            return False
        return True

    # 处理文件逻辑与其他转换函数类似
    if os.path.isfile(input_path):
        if output_path is None:
            output_path = os.path.join(os.path.dirname(input_path), f"pre_{os.path.basename(input_path)}.jsonl")
        if process_table_file(input_path, output_path, question_col, answer_col):
            processed_files += 1
            output_files.append(output_path)  # 添加这一行

    elif os.path.isdir(input_path):
        print(f"\n扫描目录：{input_path}")
        table_files = [f for f in os.listdir(input_path) if f.endswith(('.csv', '.xlsx', '.xls'))]
        print(f"找到 {len(table_files)} 个表格文件")
        
        for filename in table_files:
            filepath = os.path.join(input_path, filename)
            output_filepath = os.path.join(input_path, f"pre_{os.path.splitext(filename)[0]}.jsonl")
            if process_table_file(filepath, output_filepath, question_col, answer_col):
                processed_files += 1
                output_files.append(output_filepath)  # 添加这一行

    duration = time.time() - start_time
    print(f"\n=== 转换完成 ===")
    print(f"- 处理文件数: {processed_files}")
    print(f"- 总用时: {duration:.2f} 秒")

    # 转换完成后预览第一个输出文件的第一行
    if output_files and os.path.exists(output_files[0]):
        print("\n=== 转换结果预览 ===")
        preview_data(output_files[0], 1)  # 使用模式1预览预训练格式文件


def convert_table_to_minimind_sft(input_path, output_path=None, question_col=None, answer_col=None):
    """将表格数据转换为 MiniMind SFT 格式"""
    print("\n=== 开始转换表格数据为 MiniMind SFT 格式 ===")
    start_time = time.time()
    processed_files = 0
    output_files = []  # 记录所有输出文件路径

    def process_table_file(filepath, output_filepath, q_col, a_col):
        try:
            print(f"\n处理文件：{filepath}")
            print(f"输出文件：{output_filepath}")
            
            # 读取表格数据
            if filepath.endswith('.csv'):
                df = pd.read_csv(filepath)
            else:
                df = pd.read_excel(filepath)
            
            # 如果未指定列，使用前两列
            if q_col is None:
                q_col = df.columns[0]
            if a_col is None:
                a_col = df.columns[1]
            
            print(f"问题列：{q_col}")
            print(f"答案列：{a_col}")
            
            success_count = 0
            error_count = 0
            
            with open(output_filepath, 'w', encoding='utf-8') as outfile:
                for _, row in tqdm(df.iterrows(), total=len(df), desc="转换进度"):
                    try:
                        question = str(row[q_col]).strip()
                        answer = str(row[a_col]).strip()
                        
                        if question and answer and question.lower() != 'nan' and answer.lower() != 'nan':
                            conversations = [
                                {"role": "user", "content": question},
                                {"role": "assistant", "content": answer}
                            ]
                            output_data = {"conversations": conversations}
                            json.dump(output_data, outfile, ensure_ascii=False)
                            outfile.write('\n')
                            success_count += 1
                        else:
                            error_count += 1
                            
                    except Exception as e:
                        error_count += 1
                        print(f"处理行时出错：{e}")
                        continue
            
            print(f"\n处理完成:")
            print(f"- 成功转换: {success_count} 条")
            print(f"- 失败条数: {error_count} 条")
            
        except Exception as e:
            print(f"错误：处理文件 {filepath} 时发生错误：{e}")
            return False
        return True

    # 处理文件逻辑与其他转换函数类似
    if os.path.isfile(input_path):
        if output_path is None:
            output_path = os.path.join(os.path.dirname(input_path), f"sft_{os.path.basename(input_path)}.jsonl")
        if process_table_file(input_path, output_path, question_col, answer_col):
            processed_files += 1
            output_files.append(output_path)  # 添加这一行

    elif os.path.isdir(input_path):
        print(f"\n扫描目录：{input_path}")
        table_files = [f for f in os.listdir(input_path) if f.endswith(('.csv', '.xlsx', '.xls'))]
        print(f"找到 {len(table_files)} 个表格文件")
        
        for filename in table_files:
            filepath = os.path.join(input_path, filename)
            output_filepath = os.path.join(input_path, f"sft_{os.path.splitext(filename)[0]}.jsonl")
            if process_table_file(filepath, output_filepath, question_col, answer_col):
                processed_files += 1
                output_files.append(output_filepath)  # 添加这一行 

    duration = time.time() - start_time
    print(f"\n=== 转换完成 ===")
    print(f"- 处理文件数: {processed_files}")
    print(f"- 总用时: {duration:.2f} 秒")

    # 转换完成后预览第一个输出文件的第一行
    if output_files and os.path.exists(output_files[0]):
        print("\n=== 转换结果预览 ===")
        preview_data(output_files[0], 2)  # 使用模式2预览SFT格式文件


def convert_arrow_parquet_to_minimind_pretrain(input_path, output_path=None, question_col=None, answer_col=None):
    """将Arrow/Parquet数据转换为MiniMind预训练格式"""
    print("\n=== 开始转换Arrow/Parquet数据为MiniMind预训练格式 ===")
    start_time = time.time()
    processed_files = 0
    output_files = []  # 记录所有输出文件路径

    def process_arrow_parquet_file(filepath, output_filepath, q_col, a_col):
        try:
            print(f"\n处理文件：{filepath}")
            print(f"输出文件：{output_filepath}")
            
            # 读取Arrow/Parquet数据
            if filepath.endswith('.arrow'):
                table = pa.ipc.open_file(filepath).read_all()
                df = table.to_pandas()
            elif filepath.endswith('.parquet'):
                df = pq.read_table(filepath).to_pandas()
            else:
                print(f"不支持的文件格式：{filepath}")
                return False
            
            # 如果未指定列，使用前两列
            if q_col is None:
                q_col = df.columns[0]
            if a_col is None:
                a_col = df.columns[1]
            
            print(f"问题列：{q_col}")
            print(f"答案列：{a_col}")
            
            success_count = 0
            error_count = 0
            
            with open(output_filepath, 'w', encoding='utf-8') as outfile:
                for _, row in tqdm(df.iterrows(), total=len(df), desc="转换进度"):
                    try:
                        question = str(row[q_col]).strip()
                        answer = str(row[a_col]).strip()
                        
                        if question and answer and question.lower() != 'nan' and answer.lower() != 'nan':
                            text = f"<s>{question}{answer}</s>"
                            output_data = {"text": text}
                            json.dump(output_data, outfile, ensure_ascii=False)
                            outfile.write('\n')
                            success_count += 1
                        else:
                            error_count += 1
                            
                    except Exception as e:
                        error_count += 1
                        print(f"处理行时出错：{e}")
                        continue
            
            print(f"\n处理完成:")
            print(f"- 成功转换: {success_count} 条")
            print(f"- 失败条数: {error_count} 条")
            
        except Exception as e:
            print(f"错误：处理文件 {filepath} 时发生错误：{e}")
            return False
        return True

    # 处理文件逻辑
    if os.path.isfile(input_path):
        if output_path is None:
            output_path = os.path.join(os.path.dirname(input_path), f"pre_{os.path.splitext(os.path.basename(input_path))[0]}.jsonl")
        if process_arrow_parquet_file(input_path, output_path, question_col, answer_col):
            processed_files += 1
            output_files.append(output_path)  # 添加这一行

    elif os.path.isdir(input_path):
        print(f"\n扫描目录：{input_path}")
        arrow_parquet_files = [f for f in os.listdir(input_path) if f.endswith(('.arrow', '.parquet'))]
        print(f"找到 {len(arrow_parquet_files)} 个Arrow/Parquet文件")
        
        for filename in arrow_parquet_files:
            filepath = os.path.join(input_path, filename)
            output_filepath = os.path.join(input_path, f"pre_{os.path.splitext(filename)[0]}.jsonl")
            if process_arrow_parquet_file(filepath, output_filepath, question_col, answer_col):
                processed_files += 1
                output_files.append(output_filepath)  # 添加这一行

    duration = time.time() - start_time
    print(f"\n=== 转换完成 ===")
    print(f"- 处理文件数: {processed_files}")
    print(f"- 总用时: {duration:.2f} 秒")

    # 转换完成后预览第一个输出文件的第一行
    if output_files and os.path.exists(output_files[0]):
        print("\n=== 转换结果预览 ===")
        preview_data(output_files[0], 1)  # 使用模式1预览预训练格式文件

def convert_arrow_parquet_to_minimind_sft(input_path, output_path=None, question_col=None, answer_col=None):
    """将Arrow/Parquet数据转换为MiniMind SFT格式"""
    print("\n=== 开始转换Arrow/Parquet数据为MiniMind SFT格式 ===")
    start_time = time.time()
    processed_files = 0
    output_files = []  # 记录所有输出文件路径

    def process_arrow_parquet_file(filepath, output_filepath, q_col, a_col):
        try:
            print(f"\n处理文件：{filepath}")
            print(f"输出文件：{output_filepath}")
            
            # 读取Arrow/Parquet数据
            if filepath.endswith('.arrow'):
                table = pa.ipc.open_file(filepath).read_all()
                df = table.to_pandas()
            elif filepath.endswith('.parquet'):
                df = pq.read_table(filepath).to_pandas()
            else:
                print(f"不支持的文件格式：{filepath}")
                return False
            
            # 如果未指定列，使用前两列
            if q_col is None:
                q_col = df.columns[0]
            if a_col is None:
                a_col = df.columns[1]
            
            print(f"问题列：{q_col}")
            print(f"答案列：{a_col}")
            
            success_count = 0
            error_count = 0
            
            with open(output_filepath, 'w', encoding='utf-8') as outfile:
                for _, row in tqdm(df.iterrows(), total=len(df), desc="转换进度"):
                    try:
                        question = str(row[q_col]).strip()
                        answer = str(row[a_col]).strip()
                        
                        if question and answer and question.lower() != 'nan' and answer.lower() != 'nan':
                            conversations = [
                                {"role": "user", "content": question},
                                {"role": "assistant", "content": answer}
                            ]
                            output_data = {"conversations": conversations}
                            json.dump(output_data, outfile, ensure_ascii=False)
                            outfile.write('\n')
                            success_count += 1
                        else:
                            error_count += 1
                            
                    except Exception as e:
                        error_count += 1
                        print(f"处理行时出错：{e}")
                        continue
            
            print(f"\n处理完成:")
            print(f"- 成功转换: {success_count} 条")
            print(f"- 失败条数: {error_count} 条")
            
        except Exception as e:
            print(f"错误：处理文件 {filepath} 时发生错误：{e}")
            return False
        return True

    # 处理文件逻辑
    if os.path.isfile(input_path):
        if output_path is None:
            output_path = os.path.join(os.path.dirname(input_path), f"sft_{os.path.splitext(os.path.basename(input_path))[0]}.jsonl")
        if process_arrow_parquet_file(input_path, output_path, question_col, answer_col):
            processed_files += 1
            output_files.append(output_path)  # 添加这一行

    elif os.path.isdir(input_path):
        print(f"\n扫描目录：{input_path}")
        arrow_parquet_files = [f for f in os.listdir(input_path) if f.endswith(('.arrow', '.parquet'))]
        print(f"找到 {len(arrow_parquet_files)} 个Arrow/Parquet文件")
        
        for filename in arrow_parquet_files:
            filepath = os.path.join(input_path, filename)
            output_filepath = os.path.join(input_path, f"sft_{os.path.splitext(filename)[0]}.jsonl")
            if process_arrow_parquet_file(filepath, output_filepath, question_col, answer_col):
                processed_files += 1
                output_files.append(output_filepath)  # 添加这一行

    duration = time.time() - start_time
    print(f"\n=== 转换完成 ===")
    print(f"- 处理文件数: {processed_files}")
    print(f"- 总用时: {duration:.2f} 秒")

    # 转换完成后预览第一个输出文件的第一行
    if output_files and os.path.exists(output_files[0]):
        print("\n=== 转换结果预览 ===")
        preview_data(output_files[0], 2)  # 使用模式2预览SFT格式文件



# 新增：数据预览函数
def preview_data(file_path, mode):
    """预览数据文件的前几行"""
    try:
        print("\n=== 数据预览 ===")
        
        if mode in [1, 2] and file_path.endswith('.jsonl'):  # JSONL 模式
            with open(file_path, 'r', encoding='utf-8') as f:
                first_line = f.readline().strip()
                print("第一行数据:")
                print("-" * 40)
                data = json.loads(first_line)
                print(json.dumps(data, ensure_ascii=False, indent=2))
                print("-" * 40)
                
        elif mode in [3, 4] and file_path.endswith(('.csv', '.xlsx', '.xls')):  # 表格模式
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path, nrows=1)
            else:
                df = pd.read_excel(file_path, nrows=1)
                
            print("列名:", list(df.columns))
            print("-" * 40)
            print("第一行数据:")
            print(df.iloc[0].to_dict())
            print("-" * 40)
            
        elif mode in [5, 6] and file_path.endswith(('.arrow', '.parquet')):  # Arrow/Parquet 模式
            if file_path.endswith('.arrow'):
                table = pa.ipc.open_file(file_path).read_all()
                df = table.to_pandas().head(1)
            else:
                df = pq.read_table(file_path).to_pandas().head(1)
                
            print("列名:", list(df.columns))
            print("-" * 40)
            print("第一行数据:")
            print(df.iloc[0].to_dict())
            print("-" * 40)
            
    except Exception as e:
        print(f"预览数据时出错: {e}")

def interactive_mode():
    while True:
        print("\n=== MiniMind 数据转换工具 ===")
        print("请选择转换模式：")
        print("1. 通用sft数据格式转换为minimind预训练格式")
        print("2. 通用sft数据格式转换为minimind监督微调格式")
        print("3. 表格数据转换为minimind预训练格式")
        print("4. 表格数据转换为minimind监督微调格式")
        print("5. Arrow/Parquet数据转换为minimind预训练格式")
        print("6. Arrow/Parquet数据转换为minimind监督微调格式")
        print("0. 退出程序")
        print("=" * 40)

        try:
            mode = int(input("请输入选项 (0-6): "))
            if mode == 0:
                print("\n感谢使用，再见！")
                break
            elif mode not in [1, 2, 3, 4, 5, 6]:
                print("无效的选项，请输入 0-6 之间的数字。")
                continue
        except ValueError:
            print("无效的输入，请输入数字。")
            continue

        while True:
            input_path = input("\n请输入文件路径或目录路径：").strip()
            input_path = input_path.strip('"').strip("'")
            if input_path.lower() == 'q':
                print("\n返回主菜单...")
                break
            elif os.path.exists(input_path):
                # 新增：预览数据
                if mode in [1, 2]:  # JSONL 模式
                    if os.path.isfile(input_path) and input_path.endswith('.jsonl'):
                        try:
                            with open(input_path, 'r', encoding='utf-8') as f:
                                first_line = f.readline().strip()
                                print("\n数据预览 (第一行):")
                                print("-" * 40)
                                data = json.loads(first_line)
                                print(json.dumps(data, ensure_ascii=False, indent=2))
                                print("-" * 40)
                        except Exception as e:
                            print(f"预览数据时出错: {e}")
                
                elif mode in [3, 4]:  # 表格模式
                    if os.path.isfile(input_path):
                        try:
                            if input_path.endswith('.csv'):
                                df = pd.read_csv(input_path, nrows=1)
                            elif input_path.endswith(('.xlsx', '.xls')):
                                df = pd.read_excel(input_path, nrows=1)
                            else:
                                print("不支持的表格文件格式")
                                continue
                                
                            print("\n数据预览:")
                            print("-" * 80)
                            print("列名:", list(df.columns))
                            print("-" * 80)
                            print("第一行数据:")
                            print(df.iloc[0].to_dict())
                            print("-" * 80)
                        except Exception as e:
                            print(f"预览表格数据时出错: {e}")
                
                elif mode in [5, 6]:  # Arrow/Parquet 模式
                    if os.path.isfile(input_path):
                        try:
                            if input_path.endswith('.arrow'):
                                table = pa.ipc.open_file(input_path).read_all()
                                df = table.to_pandas().head(1)
                            elif input_path.endswith('.parquet'):
                                df = pq.read_table(input_path).to_pandas().head(1)
                            else:
                                print("不支持的文件格式")
                                continue
                                
                            print("\n数据预览:")
                            print("-" * 80)
                            print("列名:", list(df.columns))
                            print("-" * 80)
                            print("第一行数据:")
                            print(df.iloc[0].to_dict())
                            print("-" * 80)
                        except Exception as e:
                            print(f"预览Arrow/Parquet数据时出错: {e}")
                
                # 继续原有逻辑
                if mode in [3, 4, 5, 6]:  # 表格或Arrow/Parquet模式
                    question_col = input("请输入问题列名或列索引（直接回车使用第一列）：").strip() or None
                    answer_col = input("请输入答案列名或列索引（直接回车使用第二列）：").strip() or None
                    
                    if mode == 3:
                        convert_table_to_minimind_pretrain(input_path, question_col=question_col, answer_col=answer_col)
                    elif mode == 4:
                        convert_table_to_minimind_sft(input_path, question_col=question_col, answer_col=answer_col)
                    elif mode == 5:
                        convert_arrow_parquet_to_minimind_pretrain(input_path, question_col=question_col, answer_col=answer_col)
                    else:
                        convert_arrow_parquet_to_minimind_sft(input_path, question_col=question_col, answer_col=answer_col)
                else:  # JSONL 模式
                    if mode == 1:
                        convert_sft_to_minimind_pretrain(input_path)
                    else:
                        convert_sft_to_minimind_sft(input_path)
            else:
                print("路径无效，请重新输入。(输入 'q' 返回主菜单)")
                
def main():
    parser = argparse.ArgumentParser(description="转换数据为 MiniMind 训练格式")
    parser.add_argument("-i", "--input", required=False, help="输入文件路径或目录")
    parser.add_argument("-o", "--output", required=False, help="输出文件路径（可选）")
    parser.add_argument("-m", "--mode", type=int, default=0, choices=[0, 1, 2, 3, 4, 5, 6],
                        help="转换模式：0:交互模式，1:预训练格式，2:SFT格式，3:表格转预训练，4:表格转SFT，5:Arrow/Parquet转预训练，6:Arrow/Parquet转SFT")
    parser.add_argument("-q", "--question", help="问题列名或列索引（表格/Arrow/Parquet模式使用）")
    parser.add_argument("-a", "--answer", help="答案列名或列索引（表格/Arrow/Parquet模式使用）")
    parser.add_argument("--preview", action="store_true", help="预览数据（显示第一行）")

    args = parser.parse_args()
    
    print("\n=== MiniMind 数据转换工具 ===")
    if args.mode == 0:
        interactive_mode()
    elif args.mode in [1, 2, 3, 4, 5, 6]:
        if not args.input:
            print(f"错误：模式 {args.mode} 需要通过 -i 参数指定输入文件或目录。")
            return
        
        print(f"输入路径：{args.input}")
        if args.output:
            print(f"输出路径：{args.output}")
        
        # 新增：命令行模式下的数据预览
        if args.preview and os.path.isfile(args.input):
            preview_data(args.input, args.mode)
        
        if args.mode == 1:
            convert_sft_to_minimind_pretrain(args.input, args.output)
        elif args.mode == 2:
            convert_sft_to_minimind_sft(args.input, args.output)
        elif args.mode == 3:
            convert_table_to_minimind_pretrain(args.input, args.output, args.question, args.answer)
        elif args.mode == 4:
            convert_table_to_minimind_sft(args.input, args.output, args.question, args.answer)
        elif args.mode == 5:
            convert_arrow_parquet_to_minimind_pretrain(args.input, args.output, args.question, args.answer)
        elif args.mode == 6:
            convert_arrow_parquet_to_minimind_sft(args.input, args.output, args.question, args.answer)
    else:
        print("无效的模式选择")

if __name__ == "__main__":
    main()
