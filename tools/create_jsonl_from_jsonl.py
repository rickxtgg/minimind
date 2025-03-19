import json

def process_jsonl(input_file, output_file, num_lines=100000):
    """
    从指定的 JSONL 文件读取前 num_lines 行数据，并保存到一个新的 JSONL 文件。

    Args:
        input_file:  输入的 JSONL 文件路径。
        output_file: 输出的 JSONL 文件路径。
        num_lines:   要读取的行数 (默认为 10)。
    """
    try:
        with open(input_file, 'r', encoding='utf-8') as infile, \
             open(output_file, 'w', encoding='utf-8') as outfile:
            count = 0
            for line in infile:
                if count >= num_lines:
                    break
                # 解析每一行 JSON 数据
                try:
                    data = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"Error decoding line {count + 1}: {e}")
                    # 可选：根据需要决定是否继续处理或跳过错误行.  这里选择跳过
                    continue

                # 将 JSON 数据写入输出文件
                json.dump(data, outfile, ensure_ascii=False)
                outfile.write('\n')  #  JSONL 文件每行之间需要换行

                count += 1

    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == '__main__':
    input_jsonl_file = 'dataset/pretrain_hq.jsonl'  # 替换为你的输入文件
    output_jsonl_file = 'dataset/pretrain_hq_head100000.jsonl'  # 替换为你的输出文件

    process_jsonl(input_jsonl_file, output_jsonl_file)
    print(f"Successfully processed {input_jsonl_file}.  First 10 lines saved to {output_jsonl_file}")