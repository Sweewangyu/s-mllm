import transformers
from transformers import get_wsd_schedule  # 导入自定义学习率调度器
from transformers.optimization import get_cosine_with_min_lr_schedule_with_warmup  # 导入余弦学习率调度器

# 计算总训练步数
def compute_total_steps(train_dataset, training_args):
    """
    根据训练数据集和训练参数计算总的训练步数。

    参数：
    - train_dataset: 训练数据集，包含样本数量。
    - training_args: 训练参数，包含每设备批量大小、梯度累积步数和训练轮数。

    返回：
    - total_steps: 总训练步数。
    """
    total_samples = len(train_dataset)  # 数据集样本总数
    effective_batch_size = (
        training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps
    )  # 实际批量大小 = 每设备批量大小 × 梯度累积步数
    total_steps = (total_samples // effective_batch_size) * training_args.num_train_epochs
    return total_steps

# 自定义学习率调度器
def custom_lr_scheduler(optimizer, total_steps):
    """
    创建一个自定义学习率调度器，分为预热、恒定和衰减三个阶段。

    参数：
    - optimizer: 优化器，用于更新模型参数。
    - total_steps: 总的训练步数。

    返回：
    - 调度器对象。
    """
    num_warmup_steps = int(total_steps * 0)   # 预热步数为总步数的 0%
    num_stable_steps = int(total_steps * 0.2)   # 恒定阶段步数为总步数的 20%
    num_decay_steps = int(total_steps * 0.8)    # 衰减阶段步数为总步数的 80%
    print(f"Warmup Steps: {num_warmup_steps}, Stable Steps: {num_stable_steps}, Decay Steps: {num_decay_steps}")
    return get_wsd_schedule(
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_stable_steps=num_stable_steps,
        num_decay_steps=num_decay_steps,
        min_lr_ratio=0.1,  # 最小学习率比例
        num_cycles=0.5     # 半周期余弦
    )

# 余弦学习率调度器
def cos_lr_scheduler(optimizer, total_steps):
    """
    创建一个余弦学习率调度器，具有学习率预热和余弦衰减功能。

    参数：
    - optimizer: 优化器，用于更新模型参数。
    - total_steps: 总的训练步数。

    返回：
    - 调度器对象。
    """
    num_warmup_steps = int(total_steps * 0)   # 预热步数为总步数的 0%
    print(f"Warmup Steps: {num_warmup_steps}")
    return get_cosine_with_min_lr_schedule_with_warmup(
        optimizer=optimizer,
        num_training_steps=total_steps,
        num_warmup_steps=num_warmup_steps,
        min_lr_rate=0.1,  # 最小学习率比例
        num_cycles=0.5     # 半周期余弦
    )

# 自定义回调类，用于记录训练过程中的损失
class LossLoggerCallback(transformers.TrainerCallback):
    """
    训练回调类，用于记录每次日志输出中的损失值。
    """
    def __init__(self):
        self.losses = []  # 初始化一个空列表，用于存储损失值

    def on_log(self, args, state, control, logs=None, **kwargs):
        """
        在 Trainer 日志输出时被调用，如果日志中包含 'loss' 键，则将其值记录下来。

        参数：
        - args: Trainer 参数。
        - state: Trainer 的状态信息。
        - control: 控制 Trainer 的流程。
        - logs: 当前日志信息，包含损失等指标。
        """
        if 'loss' in logs:
            self.losses.append(logs['loss'])  # 记录损失值
