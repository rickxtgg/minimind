import json
import random
import re

import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import torch.distributed as dist
from sklearn.model_selection import train_test_split
import os
import ast
from tqdm import tqdm

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def Logger(content):
    # 在分布式训练中，只在主进程(rank 0)上打印日志
    if is_main_process():
        print(content)

def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0


class PretrainDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        if is_main_process():
            print(f"开始加载预训练数据: {data_path}")
        self.samples = self.load_data(data_path)
        if is_main_process():
            print(f"预训练数据加载完成，共 {len(self.samples)} 条样本")

    def load_data(self, path):
        samples = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="加载预训练数据", disable=not is_main_process()):
                data = json.loads(line.strip())
                samples.append(data)
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]

        # 构建输入文本
        text = f"{self.tokenizer.bos_token}{str(sample['text'])}{self.tokenizer.eos_token}"
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        input_ids = encoding.input_ids.squeeze()
        loss_mask = (input_ids != self.tokenizer.pad_token_id)

        X = torch.tensor(input_ids[:-1], dtype=torch.long)
        Y = torch.tensor(input_ids[1:], dtype=torch.long)
        loss_mask = torch.tensor(loss_mask[1:], dtype=torch.long)
        return X, Y, loss_mask


class SFTDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=1024, cache_size=1000000, log_interval=100000):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.jsonl_path = jsonl_path
        self.cache_size = cache_size
        self.cache = {}
        self.log_interval = log_interval
        if is_main_process():
            Logger(f"开始统计数据集大小: {jsonl_path}")
        self.file_length = self._get_file_length()
        if is_main_process():
            Logger(f"数据集大小统计完成，共 {self.file_length} 条样本")
        self.bos_id = tokenizer('<s>assistant\n', add_special_tokens=False).input_ids
        self.eos_id = tokenizer('</s>\n', add_special_tokens=False).input_ids
        self.cache_hits = 0
        self.cache_misses = 0

    def _get_file_length(self):
        count = 0
        with open(self.jsonl_path, 'r', encoding='utf-8') as f:
            for _ in tqdm(f, desc="统计数据集大小"):
                count += 1
        return count

    def __len__(self):
        return self.file_length

    def _load_line(self, index):
        # 每500000条数据输出一次加载进度
        if is_main_process() and index % 500000 == 0:
            Logger(f"正在加载数据: {index + 1}/{self.file_length} ({(index + 1)/self.file_length:.1%})")
        with open(self.jsonl_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i == index:
                    return json.loads(line.strip())
        return None

    def _manage_cache(self, index):
        # 如果缓存已满，移除最早的项
        if len(self.cache) >= self.cache_size:
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
            if is_main_process() and (self.cache_hits + self.cache_misses) % self.log_interval == 0:
                Logger(f"缓存状态 - 大小: {len(self.cache)}/{self.cache_size}, 命中率: {self.cache_hits/(self.cache_hits+self.cache_misses):.2%} (命中: {self.cache_hits}, 未命中: {self.cache_misses})")

    def _create_chat_prompt(self, conversations):
        messages = []
        for i, turn in enumerate(conversations):
            role = 'user' if i % 2 == 0 else 'assistant'
            messages.append({"role": role, "content": turn['content']})
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )

    def _generate_loss_mask(self, input_ids):
        loss_mask = [0] * len(input_ids)
        i = 0
        while i < len(input_ids):
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                for j in range(start + 1, min(end + len(self.eos_id) + 1, self.max_length)):
                    loss_mask[j] = 1
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return loss_mask

    def __getitem__(self, index):
        # 检查缓存中是否存在数据
        if index not in self.cache:
            self.cache_misses += 1
            # 加载数据并更新缓存
            self._manage_cache(index)
            self.cache[index] = self._load_line(index)
        else:
            self.cache_hits += 1

        sample = self.cache[index]
        # 构建对话提示
        prompt = self._create_chat_prompt(sample['conversations'])
        input_ids = self.tokenizer(prompt).input_ids[:self.max_length]
        input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))

        # 生成动态损失掩码
        loss_mask = self._generate_loss_mask(input_ids)

        # 构建训练数据
        X = torch.tensor(input_ids[:-1], dtype=torch.long)
        Y = torch.tensor(input_ids[1:], dtype=torch.long)
        loss_mask = torch.tensor(loss_mask[1:], dtype=torch.long)  # 对齐预测位置

        return X, Y, loss_mask


class DPODataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=4096):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        self.bos_id = tokenizer('<s>assistant\n', add_special_tokens=False).input_ids
        self.eos_id = tokenizer('</s>\n', add_special_tokens=False).input_ids
        if is_main_process():
            print(f"开始加载DPO数据: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            self.data = []
            for line in tqdm(f, desc="加载DPO数据"):
                line = line.strip()
                obj = json.loads(line)
                self.data.append(obj)
        if is_main_process():
            print(f"DPO数据加载完成，共 {len(self.data)} 条样本")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        chosen = item['chosen']  # 是一个 list，里面包含若干 {role, content}
        rejected = item['rejected']  # 同上
        chosen_prompt = self.tokenizer.apply_chat_template(
            chosen, tokenize=False, add_generation_prompt=False
        )

        rejected_prompt = self.tokenizer.apply_chat_template(
            rejected, tokenize=False, add_generation_prompt=False
        )
        chosen_encoding = self.tokenizer(
            chosen_prompt, truncation=True, max_length=self.max_length, padding='max_length'
        )
        rejected_encoding = self.tokenizer(
            rejected_prompt, truncation=True, max_length=self.max_length, padding='max_length'
        )

        chosen_input_ids = chosen_encoding['input_ids']
        chosen_loss_mask = self._generate_loss_mask(chosen_input_ids)

        rejected_input_ids = rejected_encoding['input_ids']
        rejected_loss_mask = self._generate_loss_mask(rejected_input_ids)
        x_chosen = torch.tensor(chosen_input_ids[:-1], dtype=torch.long)
        y_chosen = torch.tensor(chosen_input_ids[1:], dtype=torch.long)
        mask_chosen = torch.tensor(chosen_loss_mask[1:], dtype=torch.long)
        x_rejected = torch.tensor(rejected_input_ids[:-1], dtype=torch.long)
        y_rejected = torch.tensor(rejected_input_ids[1:], dtype=torch.long)
        mask_rejected = torch.tensor(rejected_loss_mask[1:], dtype=torch.long)

        return {
            'x_chosen': x_chosen,
            'y_chosen': y_chosen,
            'mask_chosen': mask_chosen,
            'x_rejected': x_rejected,
            'y_rejected': y_rejected,
            'mask_rejected': mask_rejected
        }

    def _generate_loss_mask(self, input_ids):
        loss_mask = [0] * len(input_ids)
        i = 0
        while i < len(input_ids):
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                for j in range(start + 1, min(end + len(self.eos_id) + 1, self.max_length)):
                    loss_mask[j] = 1
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return loss_mask


if __name__ == "__main__":
    pass
