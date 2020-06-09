from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from dataset import HANDataset
import torch
import torch.nn as nn
import time
import numpy as np
from config import Config
from tqdm import tqdm
import os
from pathlib import Path
from evaluate import evaluate
import datetime
from model.HAN import HAN

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class EarlyStopping:
    def __init__(self, patience=4):
        self.patience = patience
        self.counter = 0
        self.best_loss = np.Inf

    def __call__(self, val_loss):
        if val_loss < self.best_loss:
            early_stop = False
            get_better = True
            self.counter = 0
            self.best_loss = val_loss
        else:
            get_better = False
            self.counter += 1
            if self.counter >= self.patience:
                early_stop = True
            else:
                early_stop = False

        return early_stop, get_better


def latest_checkpoint(directory):
    if not os.path.exists(directory):
        return None
    all_checkpoints = {
        int(x.split('.')[-2].split('-')[-1]): x
        for x in os.listdir(directory)
    }
    if not all_checkpoints:
        return None
    return os.path.join(directory,
                        all_checkpoints[max(all_checkpoints.keys())])


def train():
    writer = SummaryWriter(
        log_dir=f"./runs/{datetime.datetime.now().replace(microsecond=0).isoformat()}{'-' + os.environ['REMARK'] if 'REMARK' in os.environ else ''}")

    if not os.path.exists('checkpoint'):
        os.makedirs('checkpoint')

    try:
        pretrained_word_embedding = torch.from_numpy(
            np.load('./data/train/pretrained_word_embedding.npy')).float()
    except FileNotFoundError:
        pretrained_word_embedding = None

    model = HAN(Config, pretrained_word_embedding).to(device)

    print(model)

    dataset = HANDataset('data/train/news_parsed.tsv')

    validation_size = int(Config.validation_proportion * len(dataset))
    train_size = len(dataset) - validation_size
    train_dataset, val_dataset = random_split(dataset,
                                              (train_size, validation_size))
    print(
        f"Load training dataset with train size {len(train_dataset)} and validation size {len(val_dataset)}."
    )

    train_dataloader = iter(
        DataLoader(train_dataset,
                   batch_size=Config.batch_size,
                   shuffle=True,
                   num_workers=Config.num_workers,
                   drop_last=True))

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.learning_rate)
    start_time = time.time()
    loss_full = []
    exhaustion_count = 0
    step = 0

    Path('./checkpoint').mkdir(exist_ok=True)
    if Config.load_checkpoint:
        checkpoint_path = latest_checkpoint('./checkpoint')
        if checkpoint_path is not None:
            print(f"Load saved parameters in {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            step = checkpoint['step']
            model.train()

    early_stopping = EarlyStopping()

    with tqdm(total=Config.num_batches, desc="Training") as pbar:
        for i in range(1, Config.num_batches + 1):
            try:
                minibatch = next(train_dataloader)
            except StopIteration:
                exhaustion_count += 1
                tqdm.write(
                    f"Training data exhausted for {exhaustion_count} times after {i} batches, reuse the dataset."
                )
                train_dataloader = iter(
                    DataLoader(train_dataset,
                               batch_size=Config.batch_size,
                               shuffle=True,
                               num_workers=Config.num_workers,
                               drop_last=True))
                minibatch = next(train_dataloader)

            step += 1
            # batch_size, num_(sub)categories
            y_pred = model(minibatch)
            # batch_size
            y = minibatch[Config.target]
            loss = criterion(y_pred, y.to(device))
            loss_full.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            writer.add_scalar('Train/Loss', loss.item(), step)

            if i % Config.num_batches_show_loss == 0:
                tqdm.write(
                    f"Time {time_since(start_time)}, batches {i}, current loss {loss.item():.4f}, average loss: {np.mean(loss_full):.4f}"
                )

            if i % Config.num_batches_validate == 0:
                model.eval()
                val_loss, val_report = evaluate(model, val_dataset)
                model.train()
                precision = val_report['weighted avg']['precision']
                recall = val_report['weighted avg']['recall']
                f1 = val_report['weighted avg']['f1-score']
                writer.add_scalar('Validation/loss', val_loss, step)
                writer.add_scalar('Validation/precision', precision, step)
                writer.add_scalar('Validation/recall', recall, step)
                writer.add_scalar('Validation/F1', f1, step)
                tqdm.write(
                    f"Time {time_since(start_time)}, batches {i}, validation loss: {val_loss:.4f}, validation precision: {precision:.4f}, validation recall: {recall:.4f}, validation F1: {f1:.4f}"
                )

                early_stop, get_better = early_stopping(val_loss)
                if early_stop:
                    tqdm.write('Early stop.')
                    break
                elif get_better:
                    torch.save(
                        {
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'step': step
                        }, f"./checkpoint/ckpt-{step}.pth")

            pbar.update(1)


def time_since(since):
    """
    Format elapsed time string.
    """
    now = time.time()
    elapsed_time = now - since
    return time.strftime("%H:%M:%S", time.gmtime(elapsed_time))


if __name__ == '__main__':
    print('Using device:', device)
    # torch.manual_seed(0)
    train()
