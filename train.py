import time
from copy import deepcopy

import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from tqdm.auto import tqdm

from utils import device


def get_loss_without_mask(batch, model, criterion):
    trg = batch[:, 1:].to(device)
    n_seq = batch.shape[1]
    outputs = []
    for i in range(1, n_seq):
        input = batch[:, :i]
        output = model(input.to(device))
        outputs.append(output[:, -1, :])

    output = torch.stack(outputs, dim=1)
    seq_output = output.reshape(trg.shape[0], trg.shape[1], -1)
    return criterion(seq_output.transpose(1, 2), batch[:, 1:].to(device))


def eval_model(model, loader, criterion, llama=False, masked=True):
    model.eval()
    loss_sum = 0

    for batch in loader:
        src = batch[:, :-1].to(device)
        trg = batch[:, 1:].to(device)

        with torch.no_grad():
            if masked:
                outputs = model(src)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                if llama:
                    outputs = outputs['logits']
                loss = criterion(outputs.transpose(1, 2), trg)
            else:
                loss = get_loss_without_mask(batch, model, criterion)
        loss_sum += loss.item()

    return loss_sum / len(loader)


def train(model, optimizer, loss_fun, train_loader, test_loader, n_epoch, scheduler=None, max_norm=1.0,
          max_time=(8 * 3600), llama=False, masked=True):
    losses = {'train': [], 'test': []}
    best_weights = deepcopy(model.state_dict())
    start_time = time.time()

    def train_epoch():
        model.train()

        loss_sum = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{n_epoch}"):
            src = batch[:, :-1].to(device)
            trg = batch[:, 1:].to(device)

            optimizer.zero_grad()
            if masked:
                outputs = model.forward(src)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                if llama:
                    outputs = outputs['logits']
                loss = loss_fun(outputs.transpose(1, 2), trg)
            else:
                loss = get_loss_without_mask(model, batch, loss_fun)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()

            loss_sum += loss.item()

        return loss_sum / len(train_loader)

    for epoch in range(n_epoch):
        elapsed_time = time.time() - start_time
        if elapsed_time > max_time:
            print("Training stopped: Maximum time exceeded. Loading best weights so far.")
            break

        train_loss = train_epoch()
        losses['train'].append(train_loss)

        test_loss = eval_model(model, test_loader, loss_fun, llama=llama)
        if epoch == 0 or test_loss < min(losses['test']):
            best_weights = deepcopy(model.state_dict())
        losses['test'].append(test_loss)

        print('Test loss:', round(test_loss, 4))

        if scheduler:
            scheduler.step()

    model.load_state_dict(best_weights)
    return losses


def output_losses(losses, title="Training and Validation Loss", save_filename=None, visualize=True):
    plt.figure(figsize=(10, 5))

    plt.plot(losses['train'], label='Train Loss', color='blue', linewidth=2)
    plt.plot(losses['test'], label='Test Loss', color='orange', linewidth=2)

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()

    plt.grid(True)

    if save_filename:
        plt.savefig(save_filename, format='png')

    if visualize:
        plt.show()
