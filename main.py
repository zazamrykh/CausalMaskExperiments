import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
import numpy as np
from transformers import get_cosine_schedule_with_warmup

from datasets import get_datasets, collate_batch, tokenizer
from params import DatasetTypes, Params
from utils import device, logger
from train import eval_model, output_losses, get_loss_without_mask
from network import GPTModel

from tqdm import tqdm
from tqdm.auto import tqdm
import time

from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetUtilizationRates
from pynvml import nvmlDeviceGetMemoryInfo
from copy import deepcopy

nvmlInit()
handle = nvmlDeviceGetHandleByIndex(0)

offset = 0
params = Params(dataset_type=DatasetTypes.half_seq_len, n_epoch=7, batch_size=26, learning_rate=1e-4, warmup_steps=100)
train_dataset, test_dataset = get_datasets('./data/books_txt', offset=0, dataset_type=params.dataset_type)
train_dataloader = DataLoader(train_dataset, params.batch_size, shuffle=True, collate_fn=collate_batch)
test_dataloader = DataLoader(test_dataset, params.batch_size, collate_fn=collate_batch)

vocab_size = tokenizer.vocab_size
embed_dim = 324
num_layers = 3
num_heads = 4
ff_hidden_dim = 1024
epochs = params.n_epoch 
total_training_steps = epochs * len(train_dataloader)

use_causal_mask = True  # Choose whether use mask or not
model = GPTModel(vocab_size, embed_dim, num_layers, num_heads, ff_hidden_dim, use_causal_mask=use_causal_mask).to(device)

logger.info('Using pararmeters:\n' + str(params))
total_params = sum(p.numel() for p in model.parameters())
logger.info(f"Number of parameters: {total_params}")

def train_epoch(model, callback, masked=True):
    model.train()
    losses = []
    n_inter = 0

    for batch in tqdm(train_dataloader):
        batch = batch.to(device)

        optimizer.zero_grad()

        if masked:
            assert model.use_causal_mask
            src = batch[:, :-1]  # (batch_size, seq_length - 1)
            trg = batch[:, 1:]   # (batch_size, seq_length - 1)
            logits = model(src)  # (batch_size, seq_length - 1, vocab_size)
            loss = criterion(logits.transpose(1, 2), trg)
        else:
            assert not model.use_causal_mask
            loss = get_loss_without_mask(batch, model, criterion)

        loss.backward()
        optimizer.step()
        scheduler.step()

        losses.append(loss.item())

        n_inter += 1
        if n_inter % 10000 == 0:
            callback(np.mean(losses))
            losses = []

    return np.mean(losses)

def eval_model(model, masked=True):
    model.eval()
    losses = []

    for batch in test_dataloader:
        batch = batch.to(device)

        src = batch[:, :-1]
        trg = batch[:, 1:]

        with torch.no_grad():
            if masked:
                assert model.use_causal_mask
                logits = model(src)
                loss = criterion(logits.transpose(1, 2), trg)
            else:
                assert not model.use_causal_mask
                loss = get_loss_without_mask(batch, model, criterion)
        losses.append(loss.item())

    return np.mean(losses)

optimizer = optim.AdamW(model.parameters(), params.learning_rate)
scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=params.warmup_steps,
    num_training_steps=total_training_steps
)

criterion = nn.CrossEntropyLoss()

def callback(train_loss):
    util = nvmlDeviceGetUtilizationRates(handle)
    gpu_util = util.gpu  
    mem_util = util.memory  
    mem_info = nvmlDeviceGetMemoryInfo(handle)
    mem_used = mem_info.used 
     
    eval_loss = eval_model(model, masked=True)
    model.train()
    print(f'Epoch: {epoch+1:02} | train_loss = {train_loss:.5f}, eval_loss = {eval_loss:.5f} GPU Utilization={gpu_util}%, Memory Utilization={mem_util}%, Used Memory={mem_used/(1024**2):.2f}MB')

losses = {'train': [], 'test': []}
best_loss = eval_model(model, masked=use_causal_mask)
best_weights = None

start_time = time.time()
for epoch in tqdm(range(epochs)):
    train_loss = train_epoch(model, callback, masked=use_causal_mask)
    losses['train'].append(train_loss)
    eval_loss = eval_model(model, masked=use_causal_mask)
    losses['test'].append(eval_loss)
    
    if eval_loss < best_loss:
        best_loss = eval_loss
        best_weights = deepcopy(model.state_dict())
elapsed_time = time.time() - start_time
print('Training time:', elapsed_time)

model.load_state_dict(best_weights)
print('Best loss:', best_loss)

model_state = {
    'model_state_dict': model.state_dict(),
    
    'hyperparameters': {
        'vocab_size': vocab_size,
        'embed_dim': embed_dim,
        'num_layers': num_layers,
        'num_heads': num_heads,
        'ff_hidden_dim': ff_hidden_dim,
        'lr': params.learning_rate,
        'warmup_steps': params.warmup_steps,
        'epochs': epochs
    },
    
    'notes': {
        'data': 'books_txt',
        'preprocessing': 'ernest_lab2',
        'tokenizer': 'BertTokenizerFast(data/vocab.txt)',
        'optimizer': 'AdamW',
        'scheduler': 'get_cosine_schedule_with_warmup',
        'mask': str(use_causal_mask)
    }
}

torch.save(model_state, "gpt_baseline_with_mask.pth")

output_losses(losses, save_filename='losses_masked.png', visualize=False)
