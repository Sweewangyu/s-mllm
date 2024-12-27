import transformers
from transformers import get_wsd_schedule
from transformers.optimization import get_cosine_with_min_lr_schedule_with_warmup
def compute_total_steps(train_dataset, training_args):
    total_samples = len(train_dataset)  # 数据集样本总数
    effective_batch_size = (
        training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps
    )
    total_steps = (total_samples // effective_batch_size) * training_args.num_train_epochs
    return total_steps

# 动态计算学习率调度器参数
def custom_lr_scheduler(optimizer, total_steps):
    num_warmup_steps = int(total_steps * 0)   # 预热步数为总步数的 5%
    num_stable_steps = int(total_steps * 0.2)   # 恒定阶段为总步数的 30%
    num_decay_steps = int(total_steps * 0.8)    # 衰减阶段为总步数的 60%
    print(f"Warmup Steps: {num_warmup_steps}, Stable Steps: {num_stable_steps}, Decay Steps: {num_decay_steps}")
    return get_wsd_schedule(
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_stable_steps=num_stable_steps,
        num_decay_steps=num_decay_steps,
        min_lr_ratio=0.1,  # 最小学习率比例
        num_cycles=0.5     # 半周期余弦
    )

def cos_lr_scheduler(optimizer, total_steps):
    num_warmup_steps = int(total_steps * 0)   # 预热步数为总步数的 5%
    print(f"Warmup Steps: {num_warmup_steps}")
    return get_cosine_with_min_lr_schedule_with_warmup(
        optimizer=optimizer,
        num_training_steps=total_steps,
        num_warmup_steps=num_warmup_steps,
        min_lr_rate=0.1,  # 最小学习率比例
        num_cycles=0.5     # 半周期余弦
    )


class LossLoggerCallback(transformers.TrainerCallback):
    def __init__(self):
        self.losses = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if 'loss' in logs:
            self.losses.append(logs['loss'])