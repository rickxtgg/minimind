import os
import platform
import argparse
import random
import time
import math
import warnings
import torch.distributed as dist
from contextlib import nullcontext
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer, AutoModelForCausalLM
from model.model import MiniMindLM
from model.LMConfig import LMConfig
from model.dataset import SFTDataset
from model.model_lora import *
from model.tensorboard_utils import TensorBoardLogger

# 忽略警告信息
warnings.filterwarnings('ignore')

# 日志打印函数
# 在分布式训练时只在主进程(rank 0)打印日志
def Logger(content):
    if not ddp or dist.get_rank() == 0:
        print(content)

# 学习率调度函数
# 使用余弦退火策略调整学习率
# current_step: 当前训练步数
# total_steps: 总训练步数
# lr: 初始学习率
def get_lr(current_step, total_steps, lr):
    return lr / 10 + 0.5 * lr * (1 + math.cos(math.pi * current_step / total_steps))

# 代码和full_sft「几乎」一致
# 训练一个epoch的主要逻辑
def train_epoch(epoch, wandb):
    # 使用交叉熵损失函数,reduction='none'表示不进行降维,保留每个token的损失
    loss_fct = nn.CrossEntropyLoss(reduction='none')
    start_time = time.time()
    
    for step, (X, Y, loss_mask) in enumerate(train_loader):
        # 将数据移动到指定设备
        X = X.to(args.device)  # 输入序列
        Y = Y.to(args.device)  # 目标序列
        loss_mask = loss_mask.to(args.device)  # 损失掩码,用于忽略padding token
        
        # 根据训练进度更新学习率
        lr = get_lr(epoch * iter_per_epoch + step, args.epochs * iter_per_epoch, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # 使用上下文管理器进行混合精度训练
        with ctx:
            # 前向传播
            res = model(X)
            # 计算损失
            # 1. 将logits展平为(batch_size*seq_len, vocab_size)
            # 2. 将目标序列展平为一维
            # 3. 计算每个token的交叉熵损失
            loss = loss_fct(
                res.logits.view(-1, res.logits.size(-1)),
                Y.view(-1)
            ).view(Y.size())
            # 使用loss_mask过滤掉padding token的损失
            loss = (loss * loss_mask).sum() / loss_mask.sum()
            # 加入辅助损失(如果有的话)
            loss += res.aux_loss
            # 根据梯度累积步数缩放损失
            loss = loss / args.accumulation_steps

        # 反向传播,使用梯度缩放器处理梯度
        scaler.scale(loss).backward()

        # 每accumulation_steps步进行一次参数更新
        if (step + 1) % args.accumulation_steps == 0:
            # 取消梯度缩放以进行梯度裁剪
            scaler.unscale_(optimizer)
            # 对LoRA参数进行梯度裁剪
            torch.nn.utils.clip_grad_norm_(lora_params, args.grad_clip)

            # 更新参数
            scaler.step(optimizer)
            scaler.update()

            # 清空梯度
            optimizer.zero_grad(set_to_none=True)

        # 打印训练日志
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

            # 记录wandb日志(如果启用)
            if (wandb is not None) and (not ddp or dist.get_rank() == 0):
                wandb.log({"loss": loss,
                           "lr": optimizer.param_groups[-1]['lr'],
                           "epoch_Time": spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60})

        # 定期保存模型
        if (step + 1) % args.save_interval == 0 and (not ddp or dist.get_rank() == 0):
            model.eval()
            # 保存检查点，包含LoRA权重、优化器状态和训练状态
            checkpoint = {
                'epoch': epoch,
                'step': step,
                'model_state_dict': {k: v for k, v in model.state_dict().items() if 'lora' in k},
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict() if scaler is not None else None,
                'args': args.__dict__
            }
            checkpoint_path = os.path.join(args.save_dir, 'checkpoints', f'checkpoint_{epoch}_{step}.pt')
            torch.save(checkpoint, checkpoint_path)
            # 同时保存单独的LoRA权重文件
            save_lora(model, f'{args.save_dir}/lora/{args.lora_name}_{lm_config.dim}.pth')
            model.train()

# 初始化模型和分词器
def init_model(lm_config):
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained('./model/minimind_tokenizer')
    # 初始化模型
    model = MiniMindLM(lm_config)
    # 根据是否使用MoE选择对应的checkpoint路径
    moe_path = '_moe' if lm_config.use_moe else ''
    ckp = f'./out/rlhf_{lm_config.dim}{moe_path}.pth'
    # 加载预训练权重
    state_dict = torch.load(ckp, map_location=args.device)
    # strict=False允许加载部分权重
    model.load_state_dict(state_dict, strict=False)
    return model.to(args.device), tokenizer

# 初始化分布式训练环境
def init_distributed_mode():
    if not ddp: return
    global ddp_local_rank, DEVICE

    # 初始化进程组
    dist.init_process_group(backend="nccl")
    ddp_rank = int(os.environ["RANK"])  # 全局进程序号
    ddp_local_rank = int(os.environ["LOCAL_RANK"])  # 本地进程序号
    ddp_world_size = int(os.environ["WORLD_SIZE"])  # 总进程数
    DEVICE = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(DEVICE)

if __name__ == "__main__":
    # 配置命令行参数
    parser = argparse.ArgumentParser(description="MiniMind SFT with LoRA")
    parser.add_argument("--out_dir", type=str, default="out")
    parser.add_argument("--log_dir", type=str, default="logs", help="TensorBoard日志目录")
    parser.add_argument("--epochs", type=int, default=50)  # 训练轮数
    parser.add_argument("--batch_size", type=int, default=16)  # 批次大小
    parser.add_argument("--learning_rate", type=float, default=5e-5)  # 学习率
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")  # 训练设备
    parser.add_argument("--dtype", type=str, default="bfloat16")  # 训练精度
    parser.add_argument("--use_wandb", action="store_true")  # 是否使用wandb记录训练日志
    parser.add_argument("--wandb_project", type=str, default="MiniMind-LoRA-SFT")  # wandb项目名
    parser.add_argument("--num_workers", type=int, default=1)  # 数据加载线程数
    parser.add_argument("--ddp", action="store_true")  # 是否使用分布式训练
    parser.add_argument("--accumulation_steps", type=int, default=1)  # 梯度累积步数
    parser.add_argument("--grad_clip", type=float, default=1.0)  # 梯度裁剪阈值
    parser.add_argument("--warmup_iters", type=int, default=0)  # 预热迭代次数
    parser.add_argument("--log_interval", type=int, default=100)  # 日志打印间隔
    parser.add_argument("--save_interval", type=int, default=1)  # 模型保存间隔
    parser.add_argument('--local_rank', type=int, default=-1)  # 本地进程序号
    parser.add_argument('--dim', default=512, type=int)  # 模型隐藏层维度
    parser.add_argument('--n_layers', default=8, type=int)  # 模型层数
    parser.add_argument('--max_seq_len', default=512, type=int)  # 最大序列长度
    parser.add_argument('--use_moe', default=False, type=bool)  # 是否使用MoE
    parser.add_argument("--data_path", type=str, default="./dataset/lora_identity.jsonl")  # 训练数据路径
    parser.add_argument("--lora_name", type=str, default="lora_identity", help="根据任务保存成lora_(英文/医学/心理...)")  # LoRA权重保存名称
    parser.add_argument("--resume", action="store_true", help="是否从检查点恢复训练")  # 是否从检查点恢复训练
    parser.add_argument("--checkpoint_path", type=str, default=None, help="检查点路径")  # 检查点路径
    args = parser.parse_args()

    # 初始化模型配置
    lm_config = LMConfig(dim=args.dim, n_layers=args.n_layers, max_seq_len=args.max_seq_len, use_moe=args.use_moe)
    # 创建输出目录
    args.save_dir = os.path.join(args.out_dir)
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, "checkpoints"), exist_ok=True)  # 创建检查点目录
    
    # 计算每次迭代处理的token数
    tokens_per_iter = args.batch_size * lm_config.max_seq_len
    # 设置随机种子
    torch.manual_seed(1337)
    device_type = "cuda" if "cuda" in args.device else "cpu"

    # 设置混合精度训练上下文
    ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast()
    # 检查是否为分布式训练
    ddp = int(os.environ.get("RANK", -1)) != -1
    ddp_local_rank, DEVICE = 0, "cuda:0"
    if ddp:
        init_distributed_mode()
        args.device = torch.device(DEVICE)

    # 配置wandb
    args.wandb_run_name = f"MiniMind-Lora-SFT-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"
    if args.use_wandb and (not ddp or ddp_local_rank == 0):
        import wandb
        wandb.init(project=args.wandb_project, name=args.wandb_run_name)
    else:
        wandb = None

    # 初始化TensorBoard日志记录器
    if not ddp or dist.get_rank() == 0:
        tb_logger = TensorBoardLogger(args.log_dir, f"lora_{args.lora_name}")
        tb_logger.log_hyperparameters(args)

    # 初始化模型和tokenizer
    model, tokenizer = init_model(lm_config)
    
    # 记录模型信息
    if not ddp or dist.get_rank() == 0:
        tb_logger.log_model_info(model)

    # 应用LoRA
    apply_lora(model)

    # 从检查点恢复训练
    start_epoch = 0
    if args.resume and args.checkpoint_path is not None:
        Logger(f"正在从检查点 {args.checkpoint_path} 恢复训练...")
        checkpoint = torch.load(args.checkpoint_path, map_location=args.device)
        # 加载LoRA权重
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        # 初始化优化器
        optimizer = optim.AdamW(lora_params, lr=args.learning_rate)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # 加载梯度缩放器状态
        if checkpoint['scaler_state_dict'] is not None:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        Logger(f"成功恢复训练状态，将从epoch {start_epoch}继续训练")

    # 冻结非LoRA参数
    for name, param in model.named_parameters():
        if 'lora' not in name:
            param.requires_grad = False
    # 收集LoRA参数
    lora_params = []
    for name, param in model.named_parameters():
        if 'lora' in name:
            lora_params.append(param)

    # 初始化优化器(只优化LoRA参数)
    optimizer = optim.AdamW(lora_params, lr=args.learning_rate)
    # 准备训练数据
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

    # 初始化梯度缩放器(用于混合精度训练)
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype in ['float16', 'bfloat16']))
    # 计算每个epoch的迭代次数
    iter_per_epoch = len(train_loader)

    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())  # 总参数数量
    lora_params_count = sum(p.numel() for name, p in model.named_parameters() if 'lora' in name)  # LoRA参数数量
    if not ddp or dist.get_rank() == 0:
        Logger(f"\n=== 训练配置信息 ===")
        Logger(f"模型配置:")
        Logger(f"- 总参数量: {total_params:,}")
        Logger(f"- 隐藏层维度: {lm_config.dim}")
        Logger(f"- 层数: {lm_config.n_layers}")
        Logger(f"- 最大序列长度: {lm_config.max_seq_len}")
        Logger(f"- 是否使用MoE: {lm_config.use_moe}")
        
        Logger(f"\nLoRA配置:")
        Logger(f"- LoRA参数量: {lora_params_count:,}")
        Logger(f"- LoRA参数占比: {lora_params_count / total_params * 100:.2f}%")
        Logger(f"- LoRA权重保存名称: {args.lora_name}")
        
        Logger(f"\n优化器配置:")
        Logger(f"- 学习率: {args.learning_rate}")
        Logger(f"- 梯度累积步数: {args.accumulation_steps}")
        Logger(f"- 梯度裁剪阈值: {args.grad_clip}")
        Logger(f"- 预热迭代次数: {args.warmup_iters}")
        
        Logger(f"\n训练配置:")
        Logger(f"- 训练轮数: {args.epochs}")
        Logger(f"- 批次大小: {args.batch_size}")
        Logger(f"- 训练设备: {args.device}")
        Logger(f"- 训练精度: {args.dtype}")
        Logger(f"- 是否使用分布式训练: {ddp}")
        
        Logger(f"\n数据集信息:")
        Logger(f"- 数据集路径: {args.data_path}")
        Logger(f"- 数据集大小: {len(train_ds):,} 样本")
        Logger(f"- 每轮迭代次数: {iter_per_epoch:,}")
        Logger(f"- 每次迭代处理token数: {tokens_per_iter:,}")
        
        Logger(f"\n硬件信息:")
        Logger(f"- GPU设备: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'}")
        Logger(f"- 显存使用: {torch.cuda.memory_allocated() / 1024**2:.2f}MB / {torch.cuda.get_device_properties(0).total_memory / 1024**2:.2f}MB")
        Logger("====================\n")

    # 开始训练循环
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
