# 导入必要的库和模块
import os
import platform
import argparse
import time
import math
import warnings
import pandas as pd
import torch
import torch.distributed as dist  # 分布式训练相关
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel  # DDP模型封装
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, DistributedSampler  # 数据加载和分布式采样器
from contextlib import nullcontext

from transformers import AutoTokenizer

# 导入自定义模型和配置
from model.model import MiniMindLM
from model.LMConfig import LMConfig
from model.dataset import PretrainDataset
from model.tensorboard_utils import TensorBoardLogger

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


def train_epoch(epoch, wandb):
    # 定义交叉熵损失函数，reduction='none'使得可以通过loss_mask进行加权
    loss_fct = nn.CrossEntropyLoss(reduction='none')
    start_time = time.time()
    for step, (X, Y, loss_mask) in enumerate(train_loader):
        # 将数据移动到指定设备
        X = X.to(args.device)
        Y = Y.to(args.device)
        loss_mask = loss_mask.to(args.device)

        # 根据当前步数更新学习率
        lr = get_lr(epoch * iter_per_epoch + step, args.epochs * iter_per_epoch, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # 使用混合精度训练上下文
        with ctx:
            # 前向传播
            res = model(X)
            # 计算损失：交叉熵损失 + 辅助损失（如MoE的负载均衡损失）
            loss = loss_fct(
                res.logits.view(-1, res.logits.size(-1)),
                Y.view(-1)
            ).view(Y.size())
            loss = (loss * loss_mask).sum() / loss_mask.sum()
            loss += res.aux_loss
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
                    loss.item() * args.accumulation_steps,
                    optimizer.param_groups[-1]['lr'],
                    spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60))

            # 使用wandb记录训练指标（仅在主进程中）
            if (wandb is not None) and (not ddp or dist.get_rank() == 0):
                wandb.log({"loss": loss.item() * args.accumulation_steps,
                           "lr": optimizer.param_groups[-1]['lr'],
                           "epoch_Time": spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60})

        # 定期保存模型检查点（仅在主进程中）
        if (step + 1) % args.save_interval == 0 and (not ddp or dist.get_rank() == 0):
            model.eval()
            moe_path = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/pretrain_{lm_config.dim}{moe_path}.pth'

            # 如果是DDP模型，需要获取原始模型的状态字典
            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()

            # 保存完整的检查点，包括模型、优化器和训练状态
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': state_dict,
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'lm_config': lm_config.__dict__
            }
            torch.save(checkpoint, ckp)
            model.train()


def init_model(lm_config):
    # 初始化tokenizer和模型
    tokenizer = AutoTokenizer.from_pretrained('./model/minimind_tokenizer')
    model = MiniMindLM(lm_config).to(args.device)
    Logger(f'LLM总参数量：{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f} 百万')
    return model, tokenizer


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


# 使用torchrun启动分布式训练的示例命令
# torchrun --nproc_per_node 2 1-pretrain.py
if __name__ == "__main__":
    # 配置命令行参数
    parser = argparse.ArgumentParser(description="MiniMind Pretraining")
    parser.add_argument("--out_dir", type=str, default="out")
    parser.add_argument("--log_dir", type=str, default="logs", help="TensorBoard日志目录")
    # 若要以最快速度实现zero则epochs设置为1轮；否则应当利用有限的数据训练2~6个epochs。
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16")  # 使用bfloat16进行混合精度训练
    parser.add_argument("--use_wandb", action="store_true")  # 是否使用wandb记录训练过程
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Pretrain")
    parser.add_argument("--num_workers", type=int, default=1)  # 数据加载的工作进程数
    parser.add_argument("--ddp", action="store_true")  # 是否使用分布式训练
    parser.add_argument("--accumulation_steps", type=int, default=8)  # 梯度累积步数
    parser.add_argument("--grad_clip", type=float, default=1.0)  # 梯度裁剪阈值
    parser.add_argument("--warmup_iters", type=int, default=0)  # 预热训练步数
    parser.add_argument("--log_interval", type=int, default=100)  # 日志打印间隔
    parser.add_argument("--save_interval", type=int, default=100)  # 模型保存间隔
    parser.add_argument('--local_rank', type=int, default=-1)  # 本地进程序号
    parser.add_argument('--dim', default=512, type=int)  # 模型隐藏层维度
    parser.add_argument('--n_layers', default=8, type=int)  # 模型层数
    parser.add_argument('--max_seq_len', default=512, type=int)  # 最大序列长度
    parser.add_argument('--use_moe', default=False, type=bool)  # 是否使用MoE架构
    parser.add_argument("--data_path", type=str, default="./dataset/pretrain_hq.jsonl")
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

    # 设置wandb运行名称
    args.wandb_run_name = f"MiniMind-Pretrain-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"

    # 设置混合精度训练上下文
    ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast()

    # 检测是否为分布式训练环境
    ddp = int(os.environ.get("RANK", -1)) != -1  # is this a ddp run?
    ddp_local_rank, DEVICE = 0, "cuda:0"

    # 初始化分布式训练
    if ddp:
        init_distributed_mode()
        args.device = torch.device(DEVICE)

    # 初始化wandb（仅在主进程中）
    if args.use_wandb and (not ddp or ddp_local_rank == 0):
        import wandb

        wandb.init(project=args.wandb_project, name=args.wandb_run_name)
    else:
        wandb = None

    # 初始化TensorBoard日志记录器
    if not ddp or dist.get_rank() == 0:
        tb_logger = TensorBoardLogger(args.log_dir, "pretrain")
        tb_logger.log_hyperparameters(args)

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
    Logger(f"====================================\n")

    # 初始化模型和tokenizer
    model, tokenizer = init_model(lm_config)
    
    # 记录模型信息
    if not ddp or dist.get_rank() == 0:
        tb_logger.log_model_info(model)

    # 如果指定了检查点路径，从检查点恢复训练
    if args.resume and args.checkpoint_path is not None:
        Logger(f"正在从检查点 {args.checkpoint_path} 恢复训练...")
        checkpoint = torch.load(args.checkpoint_path, map_location=args.device)
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_epoch = checkpoint['epoch']
        Logger(f"成功恢复到第 {start_epoch} 轮训练")
    else:
        start_epoch = 0

    # 创建数据集和数据加载器
    train_ds = PretrainDataset(args.data_path, tokenizer, max_length=lm_config.max_seq_len)
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
    Logger("开始训练...")
    iter_per_epoch = len(train_loader)
    for epoch in range(start_epoch, args.epochs):
        train_epoch(epoch, wandb)
        # 记录每个epoch的指标
        if not ddp or dist.get_rank() == 0:
            metrics = {"epoch": epoch}
            tb_logger.log_epoch(metrics, epoch)
    
    # 关闭TensorBoard日志记录器
    if not ddp or dist.get_rank() == 0:
        tb_logger.close()
    
    Logger("训练完成!")