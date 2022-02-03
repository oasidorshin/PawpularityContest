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

def deterministic_dataloader(seed):
    # https://pytorch.org/docs/stable/notes/randomness.html
    
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(seed)
    
    return seed_worker, g



