import os
from torch.utils.tensorboard import SummaryWriter

class TensorBoardLogger:
    def __init__(self, log_dir, model_name):
        """初始化TensorBoard日志记录器
        Args:
            log_dir: 日志保存目录
            model_name: 模型名称，用于区分不同的训练任务
        """
        self.log_dir = os.path.join(log_dir, model_name)
        os.makedirs(self.log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.log_dir)

    def log_hyperparameters(self, args):
        """记录超参数配置
        Args:
            args: 包含超参数的命名空间对象
        """
        hparams = {}
        for key, value in vars(args).items():
            if isinstance(value, (int, float, str, bool)):
                hparams[key] = value
        self.writer.add_hparams(hparams, {})

    def log_model_info(self, model):
        """记录模型信息，包括参数量等
        Args:
            model: PyTorch模型实例
        """
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.writer.add_scalar('model/total_parameters', total_params, 0)
        self.writer.add_scalar('model/trainable_parameters', trainable_params, 0)

    def log_training_step(self, metrics, global_step):
        """记录每个训练步骤的指标
        Args:
            metrics: 包含各项指标的字典
            global_step: 全局训练步数
        """
        for name, value in metrics.items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(f'train/{name}', value, global_step)

    def log_validation_step(self, metrics, global_step):
        """记录每个验证步骤的指标
        Args:
            metrics: 包含各项指标的字典
            global_step: 全局训练步数
        """
        for name, value in metrics.items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(f'val/{name}', value, global_step)

    def log_epoch(self, metrics, epoch):
        """记录每个训练轮次的指标
        Args:
            metrics: 包含各项指标的字典
            epoch: 当前轮次
        """
        for name, value in metrics.items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(f'epoch/{name}', value, epoch)

    def log_gradients_and_weights(self, model, global_step):
        """记录模型参数的梯度和权重分布
        Args:
            model: PyTorch模型实例
            global_step: 全局训练步数
        """
        for name, param in model.named_parameters():
            if param.requires_grad:
                if param.grad is not None:
                    self.writer.add_histogram(f'gradients/{name}', param.grad, global_step)
                self.writer.add_histogram(f'weights/{name}', param, global_step)

    def close(self):
        """关闭TensorBoard写入器"""
        self.writer.close()