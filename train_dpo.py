# DPO(Direct Preference Optimization)训练脚本
# 该脚本实现了基于人类偏好的模型对齐训练，主要功能：
# 1. 加载预训练模型和参考模型
# 2. 实现DPO损失计算
# 3. 支持混合精度训练和梯度累积
# 4. 实现分布式训练(DDP)
# 5. 支持wandb可视化训练过程

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
from model.dataset import DPODataset

warnings.filterwarnings('ignore')


def Logger(content):
    # 在分布式训练中，只在主进程(rank 0)上打印日志
    if not ddp or dist.get_rank() == 0:
        print(content)


def get_lr(current_step, total_steps, lr):
    # 使用余弦退火调整学习率
    # current_step: 当前训练步数
    # total_steps: 总训练步数
    # lr: 初始学习率
    return lr / 10 + 0.5 * lr * (1 + math.cos(math.pi * current_step / total_steps))


def logits_to_probs(logits, labels):
    # 将模型输出的logits转换为token级别的概率
    # logits: 模型输出的logits，形状为(batch_size, seq_len, vocab_size)
    # labels: 真实标签，形状为(batch_size, seq_len)
    # 返回: token级别的概率，形状为(batch_size, seq_len)
    log_probs = F.log_softmax(logits, dim=2)  # 对词表维度进行softmax
    probs = torch.gather(log_probs, dim=2, index=labels.unsqueeze(2)).squeeze(-1)  # 获取真实标签对应的概率
    return probs


def dpo_loss(ref_probs, probs, beta):
    # 计算DPO(Direct Preference Optimization)损失
    # ref_probs: 参考模型的token概率，形状为(batch_size, seq_len)
    # probs: 训练模型的token概率，形状为(batch_size, seq_len)
    # beta: 温度系数，用于调节loss的scale
    
    # 计算每个样本的平均概率（序列级别的概率）
    ref_probs = ref_probs.mean(dim=1)  # (batch_size,)
    probs = probs.mean(dim=1)          # (batch_size,)

    # 将chosen和rejected数据分开
    # 数据的前半部分是chosen样本，后半部分是rejected样本
    batch_size = ref_probs.shape[0]
    chosen_ref_probs = ref_probs[:batch_size // 2]    # 参考模型对chosen样本的预测概率
    reject_ref_probs = ref_probs[batch_size // 2:]    # 参考模型对rejected样本的预测概率
    chosen_probs = probs[:batch_size // 2]            # 训练模型对chosen样本的预测概率
    reject_probs = probs[batch_size // 2:]            # 训练模型对rejected样本的预测概率

    # 计算概率比值的对数差
    pi_logratios = chosen_probs - reject_probs        # 训练模型的对数概率比
    ref_logratios = chosen_ref_probs - reject_ref_probs  # 参考模型的对数概率比
    
    # 计算最终的logits并应用温度系数
    logits = pi_logratios - ref_logratios
    
    # 使用二元交叉熵作为损失函数
    # 目标是最大化chosen样本相对于rejected样本的概率
    loss = -F.logsigmoid(beta * logits)
    return loss.mean()


def train_epoch(epoch, wandb):
    # 训练一个epoch
    start_time = time.time()
    for step, batch in enumerate(train_loader):
        # 准备数据
        x_chosen = batch['x_chosen'].to(args.device)      # chosen样本的输入
        x_rejected = batch['x_rejected'].to(args.device)  # rejected样本的输入
        y_chosen = batch['y_chosen'].to(args.device)      # chosen样本的标签
        y_rejected = batch['y_rejected'].to(args.device)  # rejected样本的标签
        mask_chosen = batch['mask_chosen'].to(args.device)    # chosen样本的mask
        mask_rejected = batch['mask_rejected'].to(args.device)  # rejected样本的mask
        
        # 将chosen和rejected数据拼接在一起处理
        x = torch.cat([x_chosen, x_rejected], dim=0)
        y = torch.cat([y_chosen, y_rejected], dim=0)
        mask = torch.cat([mask_chosen, mask_rejected], dim=0)

        # 更新学习率
        lr = get_lr(epoch * iter_per_epoch + step, args.epochs * iter_per_epoch, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # 使用混合精度训练
        with ctx:
            # 首先用参考模型计算概率（不需要梯度）
            with torch.no_grad():
                ref_outputs = ref_model(x)
                ref_logits = ref_outputs.logits
            ref_probs = logits_to_probs(ref_logits, y)
            ref_probs = ref_probs * mask  # 应用mask
            
            # 然后用训练模型计算概率
            outputs = model(x)
            logits = outputs.logits
            probs = logits_to_probs(logits, y)
            probs = probs * mask  # 应用mask
            
            # 计算DPO损失
            loss = dpo_loss(ref_probs, probs, beta=0.1)
            # 根据梯度累积步数缩放损失
            loss = loss / args.accumulation_steps

        # 反向传播
        scaler.scale(loss).backward()

        # 梯度累积：每accumulation_steps步才更新一次参数
        if (step + 1) % args.accumulation_steps == 0:
            # 将梯度的scale还原回去
            scaler.unscale_(optimizer)
            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            # 更新参数
            scaler.step(optimizer)
            scaler.update()
            # 清空梯度
            optimizer.zero_grad(set_to_none=True)

        # 定期打印训练信息
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

            # 使用wandb记录训练指标（仅在主进程中）
            if (wandb is not None) and (not ddp or dist.get_rank() == 0):
                wandb.log({"loss": loss,
                           "lr": optimizer.param_groups[-1]['lr'],
                           "epoch_Time": spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60})
            
            # 使用TensorBoard记录训练指标（仅在主进程中）
            if writer is not None:
                global_step = epoch * iter_per_epoch + step
                writer.add_scalar('Training/Loss', loss.item(), global_step)
                writer.add_scalar('Training/Learning Rate', optimizer.param_groups[-1]['lr'], global_step)
                writer.add_scalar('Training/Epoch Time', spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60, global_step)
                
                # 记录梯度和参数分布
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        writer.add_histogram(f'Gradients/{name}', param.grad, global_step)
                        writer.add_histogram(f'Parameters/{name}', param, global_step)

        # 定期保存模型检查点（仅在主进程中）
        if (step + 1) % args.save_interval == 0 and (not ddp or dist.get_rank() == 0):
            model.eval()
            moe_path = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/rlhf_{lm_config.dim}{moe_path}.pth'

            # 如果是DDP模型，需要获取原始模型的状态字典
            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()

            # 保存完整的训练状态
            checkpoint = {
                'model_state_dict': state_dict,
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'epoch': epoch,
                'step': step,
                'args': args.__dict__
            }
            torch.save(checkpoint, ckp)
            model.train()


def init_model(lm_config):
    # 初始化tokenizer和模型
    tokenizer = AutoTokenizer.from_pretrained('./model/minimind_tokenizer')
    model = MiniMindLM(lm_config)
    
    # 如果指定了检查点路径且需要恢复训练
    if args.resume and args.checkpoint_path:
        Logger(f'正在从检查点 {args.checkpoint_path} 恢复训练...')
        checkpoint = torch.load(args.checkpoint_path, map_location=args.device)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        args.__dict__.update(checkpoint['args'])
        start_epoch = checkpoint['epoch']
        start_step = checkpoint['step']
        Logger(f'成功恢复到 Epoch {start_epoch}, Step {start_step}')
    else:
        # 加载SFT阶段训练好的模型权重
        moe_path = '_moe' if lm_config.use_moe else ''
        ckp = f'./out/full_sft_{lm_config.dim}{moe_path}.pth'
        state_dict = torch.load(ckp, map_location=args.device)
        model.load_state_dict(state_dict, strict=False)
    
    # 初始化参考模型（固定参数）
    ref_model = MiniMindLM(lm_config)
    ref_model.load_state_dict(state_dict, strict=False)
    ref_model.eval()
    ref_model.requires_grad_(False)

    if not ddp or dist.get_rank() == 0:
        Logger(f'LLM总参数量：{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f} 百万')
    model = model.to(args.device)
    ref_model = ref_model.to(args.device)

    return model, ref_model, tokenizer, start_epoch if args.resume else 0


def init_distributed_mode():
    # 初始化分布式训练环境
    if not ddp: return
    global ddp_local_rank, DEVICE

    # 使用nccl后端初始化进程组
    dist.init_process_group(backend="nccl")
    ddp_rank = int(os.environ["RANK"])  # 全局进程序号
    ddp_local_rank = int(os.environ["LOCAL_RANK"])  # 本地进程序号
    ddp_world_size = int(os.environ["WORLD_SIZE"])  # 总进程数
    DEVICE = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(DEVICE)


if __name__ == "__main__":
    # 配置命令行参数
    parser = argparse.ArgumentParser(description="MiniMind DPO")
    parser.add_argument("--out_dir", type=str, default="out")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=8)
    # sft阶段学习率为 「5e-6」->「5e-7」长度512，建议离线正负样本「概率」偏好对齐阶段lr <=「1e-8」长度3000，否则很容易遗忘训坏
    parser.add_argument("--learning_rate", type=float, default=1e-8)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-RLHF-SFT")
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--ddp", action="store_true")
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--warmup_iters", type=int, default=0)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=100)
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--dim', default=512, type=int)
    parser.add_argument('--n_layers', default=8, type=int)
    parser.add_argument('--max_seq_len', default=3000, type=int)
    parser.add_argument('--use_moe', default=False, type=bool)
    parser.add_argument("--resume", action="store_true", help="是否从检查点恢复训练")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="检查点文件路径")
    parser.add_argument("--data_path", type=str, default="./dataset/dpo.jsonl")

    args = parser.parse_args()

    # 初始化模型配置
    lm_config = LMConfig(dim=args.dim, n_layers=args.n_layers, max_seq_len=args.max_seq_len, use_moe=args.use_moe)
    args.save_dir = os.path.join(args.out_dir)
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)
    tokens_per_iter = args.batch_size * lm_config.max_seq_len
    torch.manual_seed(1337)  # 设置随机种子以保证可复现性
    device_type = "cuda" if "cuda" in args.device else "cpu"

   # 设置wandb运行名称
    args.wandb_run_name = f"MiniMind-Full-DPO-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"

    # 设置混合精度训练上下文
    ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast()

    # 检测是否为分布式训练环境
    ddp = int(os.environ.get("RANK", -1)) != -1  # is this a ddp run?
    ddp_local_rank, DEVICE = 0, "cuda:0"
    if ddp:
        init_distributed_mode()
        args.device = torch.device(DEVICE)

    # 初始化wandb（仅在主进程中）
    if args.use_wandb and (not ddp or ddp_local_rank == 0):
        import wandb
        wandb.init(project=args.wandb_project, name=args.wandb_run_name)
    else:
        wandb = None
        
    # 初始化TensorBoard（仅在主进程中）
    if not ddp or ddp_local_rank == 0:
        tb_log_dir = os.path.join(args.out_dir, 'tensorboard_logs', args.wandb_run_name)
        writer = SummaryWriter(log_dir=tb_log_dir)
        # 记录训练配置
        writer.add_text('Training Config/Model', f'Hidden dim: {lm_config.dim}\nLayers: {lm_config.n_layers}\nMax seq len: {lm_config.max_seq_len}\nMoE: {lm_config.use_moe}')
        writer.add_text('Training Config/Hyperparams', f'Epochs: {args.epochs}\nBatch size: {args.batch_size}\nLearning rate: {args.learning_rate}\nAccumulation steps: {args.accumulation_steps}')
    else:
        writer = None

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

    # 初始化模型和tokenizer
    model, ref_model, tokenizer = init_model(lm_config)

    # 创建数据集和数据加载器
    train_ds = DPODataset(args.data_path, tokenizer, max_length=lm_config.max_seq_len)
    train_sampler = DistributedSampler(train_ds) if ddp else None  # 分布式采样器
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        pin_memory=True,  # 将数据加载到固定内存中，加速GPU读取
        drop_last=False,
        shuffle=False,
        num_workers=args.num_workers,
        sampler=train_sampler
    )

    # 初始化混合精度训练的GradScaler
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype in ['float16', 'bfloat16']))
    # 初始化优化器
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    # 配置分布式训练模型
    if ddp:
        # 忽略位置编码参数的同步
        model._ddp_params_and_buffers_to_ignore = {"pos_cis"}
        model = DistributedDataParallel(model, device_ids=[ddp_local_rank])

    # 开始训练
    iter_per_epoch = len(train_loader)
    start_epoch = 0
    model, ref_model, tokenizer, start_epoch = init_model(lm_config)
    
    for epoch in range(start_epoch, args.epochs):
        train_epoch(epoch, wandb)
        
    # 关闭TensorBoard writer
    if writer is not None:
        writer.close()
