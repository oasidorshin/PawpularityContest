import datetime
import random, os
import numpy as np
import torch


def get_timestamp():
    timestamp = datetime.datetime.now().isoformat(' ', 'minutes')  # without seconds
    timestamp = timestamp.replace(' ', '-').replace(':', '-')
    return timestamp


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # This possibly interferes with cuda algortihms
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
