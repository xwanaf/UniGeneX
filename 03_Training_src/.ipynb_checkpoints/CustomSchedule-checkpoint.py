from torch import nn
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, BatchSampler, RandomSampler, SequentialSampler
from datasets import Dataset, load_dataset, concatenate_datasets

import math
import warnings
from functools import partial
from typing import Callable, Iterable, Optional, Tuple, Union



def _get_cosine_schedule_with_warmup_lr_lambda(
        current_step: int, *, num_warmup_steps: int, num_training_steps: int, num_cycles: float
    ):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))
        
class CustomSchedule():
    def __init__(self, warmup_ratio_or_step, total_num_batches, num_cycles = .5):
        self.warmup_steps = (
            int(total_num_batches * warmup_ratio_or_step)
            if warmup_ratio_or_step < 1
            else int(warmup_ratio_or_step)
        )
        
        self.schedule = partial(
            _get_cosine_schedule_with_warmup_lr_lambda,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=total_num_batches,
            num_cycles=num_cycles,
        )

