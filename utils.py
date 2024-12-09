import gc
import logging
import os
import random

import numpy as np
import torch


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)

logger.addHandler(console_handler)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info("Device: " + str(device))


def cleanup():
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.synchronize()


class args:
    seed = 1337


def seed_everything(seed=args.seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


seed_everything(args.seed)


current_train_number = 1

def create_dirs():
    global current_train_number
    if not os.path.exists('./runs'):
        os.makedirs('./runs')

    while os.path.exists('./runs/train' + str(current_train_number)):
        current_train_number += 1
    
    os.makedirs('./runs/train' + str(current_train_number))


def get_output_path():
    return './runs/train' + str(current_train_number) + '/'


def save_params(params, save_path, min_loss=None):
    with open(save_path, 'w') as f:
        f.write(str(params))
        
        if min_loss is not None:
            f.write('\nMinimal loss: ' + str(min_loss))
