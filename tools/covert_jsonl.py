import json
import os
import argparse
from tqdm import tqdm

def convert_sft_to_minimind_sft(input_path, output_path=None):
    """
    将通用 SFT 数据格式的 JSONL 文件转换为 MiniMind 监督微调 (SFT) 格式。

    Args:
        input_path:  输入的 JSONL 文件路径或包含 JSONL 文件的目录路径。
        output_path: 输出 JSONL 文件的路径。如果为 None（默认），
                     则输出文件名为 "converted_" + 原文件名。如果 input_path 是目录，则此参数无效。
    """

    def process_file(filepath, output_filepath):
        """处理单个 JSONL 文件。"""
        try:
            with open(filepath, 'r', encoding='utf-8') as infile, \
                 open(output_filepath, 'w', encoding='utf-8') as outfile:
                total_lines = sum(1 for _ in open(filepath, 'r', encoding='utf-8'))
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

                    except json.JSONDecodeError as e:
                        print(f"解析 JSON 行时出错（文件：{filepath}）：{e}")
                        continue
        except FileNotFoundError:
             print(f"文件未找到：{filepath}")
        except Exception as e:
            print(f"处理文件 {filepath} 时发生错误：{e}")


    if os.path.isfile(input_path):
        if output_path is None:
            output_path = "converted_" + os.path.basename(input_path)
        process_file(input_path, output_path)

    elif os.path.isdir(input_path):
        for filename in os.listdir(input_path):
            if filename.endswith(".jsonl"):
                filepath = os.path.join(input_path, filename)
                output_filepath = os.path.join(input_path, "converted_" + filename)
                process_file(filepath, output_filepath)
    else:
        print("无效的输入路径。请输入一个有效的 .jsonl 文件路径或包含 .jsonl 文件的目录。")



def convert_sft_to_minimind_pretrain(input_path, output_path=None):
    """
    将通用 SFT 数据格式的 JSONL 文件转换为 MiniMind 预训练格式。

    Args:
        input_path:  输入的 JSONL 文件路径或包含 .jsonl 文件的目录。
        output_path: 输出 JSONL 文件的路径。如果为None, 则默认为 "converted_alt_" 加原文件名
    """

    def process_file(filepath, output_filepath):
        """处理单个JSONL文件"""
        try:
            with open(filepath, 'r', encoding='utf-8') as infile, \
                    open(output_filepath, 'w', encoding='utf-8') as outfile:

                num_lines = sum(1 for _ in infile)
                infile.seek(0)

                for line in tqdm(infile, total=num_lines, desc=f"转换 {os.path.basename(filepath)}"):
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
        except FileNotFoundError:
             print(f"文件未找到：{filepath}")
        except Exception as e:
            print(f"处理文件 {filepath} 时发生错误：{e}")

    if os.path.isfile(input_path):
        if output_path is None:
            base_name, ext = os.path.splitext(input_path)
            output_path = f"{base_name}_converted_alt{ext}"
        process_file(input_path, output_path)


    elif os.path.isdir(input_path):
        for filename in os.listdir(input_path):
            if filename.endswith(".jsonl"):
                filepath = os.path.join(input_path, filename)
                output_filepath = os.path.join(input_path, "converted_alt_" + filename)
                process_file(filepath, output_filepath)
    else:
        print("无效的输入路径。请输入一个有效的 .jsonl 文件路径或包含 .jsonl 文件的目录。")


def interactive_mode():
    """交互式模式，让用户选择转换模式并输入路径。"""
    print("请选择转换模式：")
    print("1. 通用sft数据格式转换为minimind监督微调格式")
    print("2. 通用sft数据格式转换为minimind预训练格式")


    while True:
        try:
            mode = int(input("请输入选项 (1 或 2): "))
            if mode in [1, 2]:
                break
            else:
                print("无效的选项，请输入 1 或 2。")
        except ValueError:
            print("无效的输入，请输入一个数字。")

    while True:
        input_path = input("请输入 .jsonl 文件路径或包含 .jsonl 文件的目录路径：").strip()
        # 去除路径前后的引号
        input_path = input_path.strip('"').strip("'")
        if os.path.exists(input_path):
            break
        else:
            print("路径无效, 请重新输入。")

    if mode == 1:
        convert_sft_to_minimind_sft(input_path)  #  交互模式下，不指定输出路径，使用默认
    else:
        # 模式2, 不再限制仅单个文件
        convert_sft_to_minimind_pretrain(input_path)



def main():
    parser = argparse.ArgumentParser(description="转换 JSONL 文件的格式")
    parser.add_argument("-i", "--input", required=False, help="输入的 JSONL 文件路径或包含 JSONL 文件的目录")
    parser.add_argument("-o", "--output", required=False, help="输出的 JSONL 文件路径。")
    parser.add_argument("-m", "--mode", type=int, default=0, choices=[0, 1, 2],
                        help="转换模式：0 (默认) 为交互模式, 1 为通用sft数据格式转换为minimind监督微调格式，2 为通用sft数据格式转换为minimind预训练格式")

    args = parser.parse_args()

    if args.mode == 0:
        interactive_mode()
    elif args.mode == 1:
        if args.input:
             convert_sft_to_minimind_sft(args.input, args.output)
        else:
            print("错误：模式 1 需要通过 -i 参数指定输入文件或目录。")

    elif args.mode == 2:
        if args.input:
            # 模式2 不再限制
            convert_sft_to_minimind_pretrain(args.input, args.output)
        else:
            print("错误：模式 2 需要通过 -i 参数指定输入文件。")
    else:
        print("无效的模式选择")


if __name__ == "__main__":
    main()
