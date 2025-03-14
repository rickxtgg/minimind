# 推理蒸馏训练脚本
# 本脚本实现了基于思考过程的推理能力蒸馏训练，主要特点：
# 1. 通过特殊标记<think>和<answer>区分思考过程和最终答案
# 2. 对思考过程相关的token赋予更高的损失权重
# 3. 支持混合精度训练和分布式训练
# 4. 实现了动态学习率调整

import os
import platform
import argparse
import time
import math
import warnings

import pandas as pd
import torch
import torch.nn.functional as F
import torch.distributed as dist
from contextlib import nullcontext
from torch.utils.tensorboard import SummaryWriter

from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer, AutoModelForCausalLM
from model.model import MiniMindLM
from model.LMConfig import LMConfig
from model.dataset import SFTDataset

warnings.filterwarnings('ignore')


def Logger(content):
    """日志打印函数，在分布式训练时只在主进程上打印
    Args:
        content: 需要打印的内容
    """
    if not ddp or dist.get_rank() == 0:
        print(content)


def get_lr(current_step, total_steps, lr):
    """计算当前步骤的学习率，实现余弦退火调度
    Args:
        current_step: 当前训练步数
        total_steps: 总训练步数
        lr: 基础学习率
    Returns:
        当前步骤的学习率
    """
    return lr / 10 + 0.5 * lr * (1 + math.cos(math.pi * current_step / total_steps))


def train_epoch(epoch, wandb):
    """训练一个epoch
    Args:
        epoch: 当前epoch数
        wandb: wandb日志工具
    """
    # 获取特殊标记的token ID
    start_of_think_ids = tokenizer('<think>').input_ids  # 思考开始标记
    end_of_think_ids = tokenizer('</think>').input_ids   # 思考结束标记
    start_of_answer_ids = tokenizer('<answer>').input_ids  # 答案开始标记
    end_of_answer_ids = tokenizer('</answer>').input_ids   # 答案结束标记
    
    loss_fct = nn.CrossEntropyLoss(reduction='none')  # 使用不带reduction的交叉熵损失
    start_time = time.time()
    
    for step, (X, Y, loss_mask) in enumerate(train_loader):
        # 数据准备
        X = X.to(args.device)
        Y = Y.to(args.device)
        loss_mask = loss_mask.to(args.device)
        
        # 更新学习率
        lr = get_lr(epoch * iter_per_epoch + step, args.epochs * iter_per_epoch, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # 前向传播和损失计算
        with ctx:
            res = model(X)
            # 计算每个token位置的损失
            loss = loss_fct(
                res.logits.view(-1, res.logits.size(-1)),
                Y.view(-1)
            ).view(Y.size())
            
            # 识别特殊标记token的位置
            sp_ids = torch.isin(Y.view(-1),
                                torch.tensor(start_of_think_ids + end_of_think_ids
                                             + start_of_answer_ids + end_of_answer_ids
                                             ).to(args.device))
            # 对特殊标记位置的损失增加权重(10倍)
            loss_mask = loss_mask.view(-1)
            loss_mask_sum = loss_mask.sum()
            loss_mask[sp_ids] = 10
            loss_mask = loss_mask.view(Y.size())
            
            # 计算加权平均损失
            loss = (loss * loss_mask).sum() / loss_mask_sum
            if hasattr(res, 'aux_loss'):  # 如果使用MoE，加入辅助损失
                loss += res.aux_loss
            loss = loss / args.accumulation_steps  # 梯度累积

        # 反向传播
        scaler.scale(loss).backward()

        # 梯度累积和更新
        if (step + 1) % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            scaler.step(optimizer)
            scaler.update()

            optimizer.zero_grad(set_to_none=True)

        # 日志记录
        if step % args.log_interval == 0:
            spend_time = time.time() - start_time
            Logger(
                'Epoch:[{}/{}]({}/{}) loss:{:.3f} lr:{:.12f} epoch_Time:{}min:'.format(
                    epoch + 1,
                    args.epochs,
                    step,
                    iter_per_epoch,
                    loss.item(),
                    optimizer.param_groups[-1]['lr'],
                    spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60))

            if (wandb is not None) and (not ddp or dist.get_rank() == 0):
                wandb.log({"loss": loss,
                           "lr": optimizer.param_groups[-1]['lr'],
                           "epoch_Time": spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60})

        # 模型保存
        if (step + 1) % args.save_interval == 0 and (not ddp or dist.get_rank() == 0):
            model.eval()
            moe_path = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/reason_{lm_config.dim}{moe_path}.pth'
            
            checkpoint = {
                'epoch': epoch,
                'step': step,
                'model_state_dict': model.module.state_dict() if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'args': args
            }
            torch.save(checkpoint, ckp)
            model.train()


def init_model(lm_config):
    """初始化模型和分词器
    Args:
        lm_config: 模型配置
    Returns:
        model: 初始化好的模型
        tokenizer: 分词器
    """
    tokenizer = AutoTokenizer.from_pretrained('./model/minimind_tokenizer')
    model = MiniMindLM(lm_config)
    moe_path = '_moe' if lm_config.use_moe else ''
    ckp = f'./out/rlhf_{lm_config.dim}{moe_path}.pth'
    state_dict = torch.load(ckp, map_location=args.device)
    model.load_state_dict(state_dict, strict=False)
    Logger(f'LLM总参数量：{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f} 百万')
    model = model.to(args.device)
    return model, tokenizer


def init_distributed_mode():
    """初始化分布式训练环境
    设置进程组、设备和进程间通信后端
    """
    if not ddp: return
    global ddp_local_rank, DEVICE

    dist.init_process_group(backend="nccl")
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    DEVICE = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(DEVICE)


if __name__ == "__main__":
    # 配置命令行参数
    parser = argparse.ArgumentParser(description="MiniMind Distill Reasoning")
    parser.add_argument("--out_dir", type=str, default="out")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Full-SFT")
    parser.add_argument("--tensorboard_dir", type=str, default="runs", help="TensorBoard日志目录")
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--ddp", action="store_true")
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--warmup_iters", type=int, default=0)
    parser.add_argument("--log_interval", type=int, default=1)
    parser.add_argument("--save_interval", type=int, default=50)
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--dim', default=512, type=int)
    parser.add_argument('--n_layers', default=8, type=int)
    parser.add_argument('--max_seq_len', default=1024, type=int)
    parser.add_argument('--use_moe', default=False, type=bool)
    parser.add_argument("--data_path", type=str, default="./dataset/r1_mix_1024.jsonl")
    parser.add_argument("--resume", action="store_true", help="是否从检查点恢复训练")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="检查点路径")

    args = parser.parse_args()

    # 初始化模型配置
    lm_config = LMConfig(dim=args.dim, n_layers=args.n_layers, max_seq_len=args.max_seq_len, use_moe=args.use_moe)
    args.save_dir = os.path.join(args.out_dir)
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)
    tokens_per_iter = args.batch_size * lm_config.max_seq_len
    torch.manual_seed(1337)  # 设置随机种子以保证可复现性
    device_type = "cuda" if "cuda" in args.device else "cpu"

    args.wandb_run_name = f"MiniMind-Distill-Reasoning-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"

    # 设置混合精度训练上下文
    ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast()
    ddp = int(os.environ.get("RANK", -1)) != -1  # is this a ddp run?
    ddp_local_rank, DEVICE = 0, "cuda:0"
    if ddp:
        init_distributed_mode()
        args.device = torch.device(DEVICE)

    # 初始化wandb日志
    if args.use_wandb and (not ddp or ddp_local_rank == 0):
        import wandb
        wandb.init(project=args.wandb_project, name=args.wandb_run_name)
    else:
        wandb = None

    # 初始化TensorBoard
    if not ddp or ddp_local_rank == 0:
        tb_writer = SummaryWriter(os.path.join(args.tensorboard_dir, args.wandb_run_name))
        # 记录配置信息
        tb_writer.add_text('Config/Model', f'Hidden dim: {lm_config.dim}, Layers: {lm_config.n_layers}, Max seq len: {lm_config.max_seq_len}, MoE: {lm_config.use_moe}')
        tb_writer.add_text('Config/Training', f'Epochs: {args.epochs}, Batch size: {args.batch_size}, Learning rate: {args.learning_rate}, Device: {args.device}, Dtype: {args.dtype}')
    else:
        tb_writer = None

    # 记录训练配置信息
    Logger("========== 训练配置信息 ==========")
    Logger(f"模型配置:")
    Logger(f"  - 隐藏层维度: {lm_config.dim}")
    Logger(f"  - 层数: {lm_config.n_layers}")
    Logger(f"  - 最大序列长度: {lm_config.max_seq_len}")
    Logger(f"  - 是否使用MoE: {lm_config.use_moe}")
    Logger(f"\n训练参数:")
    Logger(f"  - 训练轮数: {args.epochs}")
    Logger(f"  - 批次大小: {args.batch_size}")
    Logger(f"  - 学习率: {args.learning_rate}")
    Logger(f"  - 设备: {args.device}")
    Logger(f"  - 数据类型: {args.dtype}")
    Logger(f"  - 梯度累积步数: {args.accumulation_steps}")
    Logger(f"  - 梯度裁剪阈值: {args.grad_clip}")
    Logger(f"  - 是否使用分布式训练: {args.ddp}")
    Logger(f"\n数据配置:")
    Logger(f"  - 数据路径: {args.data_path}")
    Logger(f"  - 每步token数: {tokens_per_iter}")
    Logger(f"  - 是否从检查点恢复: {args.resume}")
    if args.resume:
        Logger(f"  - 检查点路径: {args.checkpoint_path}")
    Logger(f"====================================\n")

    # 初始化模型和数据加载器
    model, tokenizer = init_model(lm_config)
    
    # 从检查点恢复
    start_epoch = 0
    if args.resume and args.checkpoint_path is not None:
        Logger(f'从检查点恢复训练: {args.checkpoint_path}')
        checkpoint = torch.load(args.checkpoint_path, map_location=args.device)
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_epoch = checkpoint['epoch']
        Logger(f'成功恢复到Epoch {start_epoch}')

    train_ds = SFTDataset(args.data_path, tokenizer, max_length=lm_config.max_seq_len)
    train_sampler = DistributedSampler(train_ds) if ddp else None
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        pin_memory=True,
        drop_last=False,
        shuffle=False,
        num_workers=args.num_workers,
        sampler=train_sampler
    )

    # 初始化优化器和混合精度训练
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype in ['float16', 'bfloat16']))
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    # 分布式训练设置
    if ddp:
        model._ddp_params_and_buffers_to_ignore = {"pos_cis"}
        model = DistributedDataParallel(model, device_ids=[ddp_local_rank])

    # 开始训练
    Logger("开始训练...")
    iter_per_epoch = len(train_loader)
    for epoch in range(start_epoch, args.epochs):
        train_epoch(epoch, wandb)
    
    Logger("训练完成!")
