import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler

import transformers

import timeit, logging

from tqdm import tqdm

from utils import deterministic_dataloader


# https://github.com/facebookresearch/mixup-cifar10/blob/main/train.py
def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    mixed_y = lam * y + (1 - lam) * y[index]
    return mixed_x, mixed_y, lam


def train_epoch(model,
                device,
                dataloader,
                loss_fn,
                optimizer,
                batch_scheduler=None,
                metric=None,
                mixup=0,
                clip_grad_norm=0):
    train_loss = torch.tensor(0.0, device=device)
    train_metric = torch.tensor(0.0, device=device)
    model.train()

    for id_, X, target in tqdm(dataloader):
        X, target = X.to(device), target.to(device)

        optimizer.zero_grad()

        if mixup:
            X_mixup, target_mixup, lam = mixup_data(X, target, alpha=mixup)
            output = model(X_mixup)
            loss = loss_fn(output, target_mixup)
        else:
            output = model(X)
            loss = loss_fn(output, target)

        loss.backward()

        if clip_grad_norm:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)

        optimizer.step()

        if batch_scheduler is not None:
            batch_scheduler.step()

        if metric is not None:
            with torch.no_grad():
                train_metric += metric(output, target).detach() * X.size(0)

        train_loss += loss.detach() * X.size(0)

    if metric is not None:
        return train_loss.item() / len(dataloader.sampler), train_metric.item() / len(dataloader.sampler)
    else:
        return train_loss.item() / len(dataloader.sampler)


def valid_epoch(model, device, dataloader, loss_fn):
    valid_loss = torch.tensor(0.0, device=device)

    model.eval()

    for id_, X, target in tqdm(dataloader):
        with torch.no_grad():
            X, target = X.to(device), target.to(device)

            output = model(X)
            loss = loss_fn(output, target)

            valid_loss += loss.detach() * X.size(0)

    return valid_loss.item() / len(dataloader.sampler)


def train_loop(model_class,
               opt_criterion,
               val_criterion,
               train_dataset,
               val_dataset,
               splits,
               folds,
               foldername,
               args,
               train_epoch=train_epoch,
               valid_epoch=valid_epoch):
    device = args.device
    val_losses = []

    for fold, (train_idx, val_idx) in enumerate(splits):
        if fold not in folds:
            continue

        logging.info(f"\n Fold {fold + 1} \n")

        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)

        seed_worker, g = deterministic_dataloader(args.seed)

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler,
                                  num_workers=args.num_workers, pin_memory=True, worker_init_fn=seed_worker, generator=g)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, sampler=val_sampler,
                                num_workers=args.num_workers, pin_memory=True, worker_init_fn=seed_worker, generator=g)

        model = model_class(name=args.model_name, n_classes=args.n_classes, pretrained=args.pretrained)
        model.to(device)

        optimizer = transformers.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        # FastAI params for AdamW, larger eps -> closer to sgd, can tune
        # optimizer = transformers.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas = (0.9, 0.99), eps = 1e-05)

        batch_scheduler = transformers.get_cosine_schedule_with_warmup(optimizer,
                                                                       num_warmup_steps=(args.epochs * len(
                                                                           train_loader)) * args.warmup_coef,
                                                                       num_training_steps=args.epochs * len(
                                                                           train_loader))

        for epoch in range(args.epochs):
            start_time = timeit.default_timer()
            train_loss, train_metric = train_epoch(model, device, train_loader, opt_criterion,
                                                   optimizer,
                                                   batch_scheduler=batch_scheduler,
                                                   metric=val_criterion,
                                                   mixup=args.mixup,
                                                   clip_grad_norm=args.clip_grad_norm)
            val_loss = valid_epoch(model, device, val_loader, val_criterion)
            end_time = timeit.default_timer()

            total = end_time - start_time

            train_loss = np.mean(np.array(train_loss))
            val_loss = np.sqrt(np.mean(np.array(val_loss)))
            train_metric = np.sqrt(np.mean(np.array(train_metric)))

            logging.info(
                f"Epoch: {epoch + 1} | T loss: {train_loss:.4f} T metr: {train_metric:.4f} V metr: {val_loss:.4f} Time: {total:.4f}")

        val_losses.append(val_loss)
        torch.save(model.state_dict(), f"models/{foldername}/fold_{fold}.pth")

    logging.info(f"\n Avg final val loss {np.sqrt(np.mean(np.array(val_losses) ** 2))}, {np.std(val_losses)} \n")
