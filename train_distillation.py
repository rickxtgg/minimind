# 知识蒸馏训练脚本
# 本脚本实现了基于知识蒸馏的模型训练过程，将大模型(教师模型)的知识迁移到小模型(学生模型)中
# 主要特点：
# 1. 支持混合精度训练
# 2. 支持分布式训练(DDP)
# 3. 结合了交叉熵损失和知识蒸馏损失
# 4. 实现了动态学习率调整

import os
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


def distillation_loss_fn(student_logits, teacher_logits, temperature=1.0, reduction='batchmean'):
    """计算知识蒸馏损失（KL散度）
    Args:
        student_logits: 学生模型的输出logits
        teacher_logits: 教师模型的输出logits
        temperature: 温度参数，用于软化概率分布
        reduction: 降维方式
    Returns:
        知识蒸馏损失
    """
    with torch.no_grad():
        teacher_probs = F.softmax(teacher_logits / temperature, dim=-1).detach()

    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)

    kl = F.kl_div(
        student_log_probs,
        teacher_probs,
        reduction=reduction
    )
    return (temperature ** 2) * kl


def train_epoch(epoch, wandb, alpha=0.0, temperature=1.0):
    """训练一个epoch
    Args:
        epoch: 当前epoch数
        wandb: wandb日志工具
        alpha: 损失权重系数，用于平衡CE损失和蒸馏损失
        temperature: 知识蒸馏的温度参数
    """
    start_time = time.time()

    # 确保教师模型处于评估模式且不计算梯度
    if teacher_model is not None:
        teacher_model.eval()
        teacher_model.requires_grad_(False)

    for step, (X, Y, loss_mask) in enumerate(train_loader):
        # 数据准备
        X = X.to(args.device)
        Y = Y.to(args.device)
        loss_mask = loss_mask.to(args.device)
        
        # 更新学习率
        lr = get_lr(epoch * iter_per_epoch + step,
                    args.epochs * iter_per_epoch,
                    args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # 前向传播（学生模型）
        with ctx:
            res = model(X)
            student_logits = res.logits

        # 教师模型前向传播（只在eval & no_grad）
        if teacher_model is not None:
            with torch.no_grad():
                teacher_logits = teacher_model(X).logits
                vocab_size_student = student_logits.size(-1)  # N
                teacher_logits = teacher_logits[..., :vocab_size_student]

        # ========== 计算损失 ==========
        # 1) Ground-Truth CE Loss（可选）
        loss_mask_flat = loss_mask.view(-1)
        ce_loss = F.cross_entropy(
            student_logits.view(-1, student_logits.size(-1)),
            Y.view(-1),
            ignore_index=0,
            reduction='none'
        )
        ce_loss = torch.sum(ce_loss * loss_mask_flat) / loss_mask_flat.sum()
        if lm_config_student.use_moe:
            ce_loss += res.aux_loss

        # 2) Distillation Loss（可选）
        if teacher_model is not None:
            # 只在有效token位置做蒸馏
            distill_loss = distillation_loss_fn(
                student_logits.view(-1, student_logits.size(-1))[loss_mask_flat == 1],
                teacher_logits.view(-1, teacher_logits.size(-1))[loss_mask_flat == 1],
                temperature=temperature
            )
        else:
            distill_loss = torch.tensor(0.0, device=args.device)

        # 3) 总损失 = alpha * CE + (1-alpha) * Distill
        loss = alpha * ce_loss + (1 - alpha) * distill_loss

        # 反向传播和优化
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
                'Epoch:[{}/{}]({}/{}) loss:{:.4f} lr:{:.12f} epoch_Time:{}min:'.format(
                    epoch,
                    args.epochs - 1,
                    step,
                    iter_per_epoch,
                    loss.item(),
                    optimizer.param_groups[-1]['lr'],
                    spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60
                )
            )

            if (wandb is not None) and (not ddp or dist.get_rank() == 0):
                wandb.log({
                    "loss": loss.item(),
                    "ce_loss": ce_loss.item(),
                    "distill_loss": distill_loss.item() if teacher_model is not None else 0.0,
                    "lr": optimizer.param_groups[-1]['lr'],
                    "last-time": spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60
                })
                
                # 记录到TensorBoard
                if tb_writer is not None:
                    global_step = epoch * iter_per_epoch + step
                    tb_writer.add_scalar('Loss/total', loss.item(), global_step)
                    tb_writer.add_scalar('Loss/ce', ce_loss.item(), global_step)
                    tb_writer.add_scalar('Loss/distill', distill_loss.item() if teacher_model is not None else 0.0, global_step)
                    tb_writer.add_scalar('Training/learning_rate', optimizer.param_groups[-1]['lr'], global_step)
                    tb_writer.add_scalar('Training/epoch_time_remaining', spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60, global_step)
                    
                    # 记录梯度和参数分布
                    for name, param in model.named_parameters():
                        if param.grad is not None:
                            tb_writer.add_histogram(f'Gradients/{name}', param.grad, global_step)
                            tb_writer.add_histogram(f'Parameters/{name}', param.data, global_step)

        # 模型保存
        if (step + 1) % args.save_interval == 0 and (not ddp or dist.get_rank() == 0):
            model.eval()
            moe_path = '_moe' if lm_config_student.use_moe else ''
            ckp = f'{args.save_dir}/full_dist_{lm_config_student.dim}{moe_path}.pth'
            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()
            
            # 保存完整的检查点
            checkpoint = {
                'epoch': epoch,
                'student_model': state_dict,
                'teacher_model': teacher_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scaler': scaler.state_dict(),
                'args': args
            }
            torch.save(checkpoint, ckp)
            model.train()


def init_student_model(lm_config):
    """初始化学生模型
    Args:
        lm_config: 模型配置
    Returns:
        model: 初始化好的模型
        tokenizer: 分词器
    """
    tokenizer = AutoTokenizer.from_pretrained('./model/minimind_tokenizer')
    model = MiniMindLM(lm_config)
    moe_path = '_moe' if lm_config.use_moe else ''
    ckp = f'./out/full_sft_{lm_config.dim}{moe_path}.pth'
    state_dict = torch.load(ckp, map_location=args.device)
    model.load_state_dict(state_dict, strict=False)
    Logger(f'学生模型(LLM)总参数量：{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f} 百万')
    model = model.to(args.device)

    return model, tokenizer


def init_teacher_model(lm_config):
    """初始化教师模型
    Args:
        lm_config: 模型配置
    Returns:
        model: 初始化好的模型
    """
    model = MiniMindLM(lm_config)
    moe_path = '_moe' if lm_config.use_moe else ''
    ckp = f'./out/full_sft_{lm_config.dim}{moe_path}.pth'
    state_dict = torch.load(ckp, map_location=args.device)
    model.load_state_dict(state_dict, strict=False)
    Logger(f'教师模型(LLM)总参数量：{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f} 百万')
    model = model.to(args.device)
    return model


def init_distributed_mode():
    """初始化分布式训练环境"""
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
    parser = argparse.ArgumentParser(description="MiniMind Knowledge Distillation")
    parser.add_argument("--out_dir", type=str, default="out")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
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
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=100)
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument("--data_path", type=str, default="./dataset/sft_data.jsonl")
    parser.add_argument("--resume", action="store_true", help="从检查点恢复训练")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="检查点路径")

    args = parser.parse_args()
    # 定义学生模型和教师模型
    lm_config_student = LMConfig(dim=512, n_layers=8, max_seq_len=512)  # 小模型配置
    lm_config_teacher = LMConfig(dim=768, n_layers=16, max_seq_len=512)  # 大模型配置
    max_seq_len = lm_config_student.max_seq_len
    args.save_dir = os.path.join(args.out_dir)
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)
    tokens_per_iter = args.batch_size * max_seq_len
    torch.manual_seed(1337)  # 设置随机种子以保证可复现性
    device_type = "cuda" if "cuda" in args.device else "cpu"

    args.wandb_run_name = f"MiniMind-Dist-SFT-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"

    # 设置混合精度训练上下文
    ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast()
    ddp = int(os.environ.get("RANK", -1)) != -1  # is this a ddp run?
    ddp_local_rank, DEVICE = 0, "cuda:0"
    if ddp:
        init_distributed_mode()
        args.device = torch.device(DEVICE)

    if args.use_wandb and (not ddp or ddp_local_rank == 0):
        import wandb
        wandb.init(project=args.wandb_project, name=args.wandb_run_name)
    else:
        wandb = None

    # 初始化TensorBoard
    if not ddp or ddp_local_rank == 0:
        tb_writer = SummaryWriter(os.path.join(args.tensorboard_dir, args.wandb_run_name))
        # 记录配置信息
        tb_writer.add_text('Config/Student Model', f'Hidden dim: {lm_config_student.dim}, Layers: {lm_config_student.n_layers}, Max seq len: {lm_config_student.max_seq_len}, MoE: {lm_config_student.use_moe}')
        tb_writer.add_text('Config/Teacher Model', f'Hidden dim: {lm_config_teacher.dim}, Layers: {lm_config_teacher.n_layers}, Max seq len: {lm_config_teacher.max_seq_len}, MoE: {lm_config_teacher.use_moe}')
        tb_writer.add_text('Config/Training', f'Epochs: {args.epochs}, Batch size: {args.batch_size}, Learning rate: {args.learning_rate}, Device: {args.device}, Dtype: {args.dtype}')
    else:
        tb_writer = None


    # 记录训练配置信息
    Logger("========== 训练配置信息 ==========")
    Logger(f"学生模型配置:")
    Logger(f"  - 隐藏层维度: {lm_config_student.dim}")
    Logger(f"  - 层数: {lm_config_student.n_layers}")
    Logger(f"  - 最大序列长度: {lm_config_student.max_seq_len}")
    Logger(f"  - 是否使用MoE: {lm_config_student.use_moe}")
    Logger(f"\n教师模型配置:")
    Logger(f"  - 隐藏层维度: {lm_config_teacher.dim}")
    Logger(f"  - 层数: {lm_config_teacher.n_layers}")
    Logger(f"  - 最大序列长度: {lm_config_teacher.max_seq_len}")
    Logger(f"  - 是否使用MoE: {lm_config_teacher.use_moe}")
    Logger(f"\n训练参数:")
    Logger(f"  - 训练轮数: {args.epochs}")
    Logger(f"  - 批次大小: {args.batch_size}")
    Logger(f"  - 学习率: {args.learning_rate}")
    Logger(f"  - 设备: {args.device}")
    Logger(f"  - 数据类型: {args.dtype}")
    Logger(f"  - 梯度累积步数: {args.accumulation_steps}")
    Logger(f"  - 梯度裁剪阈值: {args.grad_clip}")
    Logger(f"  - 是否使用分布式训练: {args.ddp}")
    Logger(f"  - 蒸馏温度: {args.temperature}")
    Logger(f"  - 蒸馏损失权重: {args.alpha}")
    Logger(f"\n数据配置:")
    Logger(f"  - 数据路径: {args.data_path}")
    Logger(f"  - 每步token数: {tokens_per_iter}")
    Logger(f"  - 是否从检查点恢复: {args.resume}")
    if args.resume:
        Logger(f"  - 检查点路径: {args.checkpoint_path}")
    Logger(f"====================================\n")

    # 初始化学生模型和教师模型
    model, tokenizer = init_student_model(lm_config_student)
    teacher_model = init_teacher_model(lm_config_teacher)

    # 从检查点恢复训练
    start_epoch = 0
    if args.resume and args.checkpoint_path is not None:
        Logger(f"正在从检查点 {args.checkpoint_path} 恢复训练...")
        checkpoint = torch.load(args.checkpoint_path, map_location=args.device)
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model.module.load_state_dict(checkpoint['student_model'])
        else:
            model.load_state_dict(checkpoint['student_model'])
        teacher_model.load_state_dict(checkpoint['teacher_model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scaler.load_state_dict(checkpoint['scaler'])
        start_epoch = checkpoint['epoch'] + 1
        Logger(f"成功恢复到 Epoch {start_epoch}")

    train_ds = SFTDataset(args.data_path, tokenizer, max_length=max_seq_len)
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

    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype in ['float16', 'bfloat16']))
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    if ddp:
        model._ddp_params_and_buffers_to_ignore = {"pos_cis"}
        model = DistributedDataParallel(model, device_ids=[ddp_local_rank])

    iter_per_epoch = len(train_loader)
    for epoch in range(start_epoch, args.epochs):
        train_epoch(epoch, wandb)
