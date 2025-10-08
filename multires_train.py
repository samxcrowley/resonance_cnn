import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from multires_numlevels_cnn import MultiRes_NumLevels_CNN
from multires_numlevels_cnn import MultiRes_NumLevels_SmallCNN
import preprocessing
import load_data
import sys
import os

SEED = 22

training_path = f'data/o16/o16_training_small.gz'
images_path = f'data/images.pt'

num_workers = 32
subset_size = 2000

cropping_strength = 0.1

# training
num_epochs = 150
batch_size = 32
lr = 1e-3
weight_decay = 1e-4

# model params.
dropout_p = 0.3
base = 40
kernel_size = 3
gradients = True

crit = torch.nn.SmoothL1Loss()

def train_epoch(net, loader, optimizer, device, grad_clip=None):

    net.train()

    running = {
        'loss': 0.0, 
        'mae': 0.0,
        'acc_exact': 0,
        'acc_within_1': 0,
        'count': 0
    }

    for batch_images, batch_target_params, batch_target_masks in loader:

        batch_images = batch_images.to(device=device, dtype=torch.float32)
        batch_target_params = batch_target_params.to(device=device, dtype=torch.float32)
        batch_target_masks = batch_target_masks.to(device=device, dtype=torch.bool)

        num_target = batch_target_masks.sum(dim=1).to(device=device, dtype=torch.float32)

        num_pred = net(batch_images)

        loss = crit(num_pred, num_target)

        optimizer.zero_grad(set_to_none=True)

        loss.backward()

        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(net.parameters(), grad_clip)
        optimizer.step()

        with torch.no_grad():
            batch_size = batch_images.size(0)
            running['loss'] += loss.item() * batch_size
            
            # round predictions for discrete evaluation
            preds_rounded = torch.round(num_pred)
            
            running['mae'] += torch.abs(num_pred - num_target).sum().item()
            running['acc_exact'] += (preds_rounded == num_target).sum().item()
            running['acc_within_1'] += (torch.abs(preds_rounded - num_target) <= 1).sum().item()
            running['count'] += batch_size

    n = running['count']

    metrics = {
        'loss': running['loss'] / n,
        'mae': running['mae'] / n,
        'acc_exact': running['acc_exact'] / n,
        'acc_within_1': running['acc_within_1'] / n
    }

    return metrics
    
def eval_epoch(net, loader, device):

    net.eval()
    
    running = {
        'loss': 0.0, 
        'mae': 0.0,
        'acc_exact': 0,
        'acc_within_1': 0,
        'count': 0
    }

    with torch.no_grad():
        for batch_images, batch_target_params, batch_target_masks in loader:

            batch_images = batch_images.to(device=device, dtype=torch.float32)
            batch_target_params = batch_target_params.to(device=device, dtype=torch.float32)
            batch_target_masks = batch_target_masks.to(device=device, dtype=torch.bool)

            num_target = batch_target_masks.sum(dim=1).to(device=device, dtype=torch.float32)

            num_pred = net(batch_images)

            loss = crit(num_pred, num_target)

            batch_size = batch_images.size(0)
            running['loss'] += loss.item() * batch_size
            
            # round predictions for discrete evaluation
            preds_rounded = torch.round(num_pred)
            
            running['mae'] += torch.abs(num_pred - num_target).sum().item()
            running['acc_exact'] += (preds_rounded == num_target).sum().item()
            running['acc_within_1'] += (torch.abs(preds_rounded - num_target) <= 1).sum().item()
            running['count'] += batch_size

    n = running['count']

    metrics = {
        'loss': running['loss'] / n,
        'mae': running['mae'] / n,
        'acc_exact': running['acc_exact'] / n,
        'acc_within_1': running['acc_within_1'] / n
    }

    return metrics
    
def main():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    images, target_params, target_masks = load_data.load_images_and_targets('multi', cropping_strength)

    # if we have defined a smaller subset, cut off the unneeded samples
    if subset_size < len(images):
        images = images[:subset_size]
        target_params = target_params[:subset_size]
        target_masks = target_masks[:subset_size]

    print(f'Images shape: {images.shape}')
    print(f'Targets shape: {target_params.shape}, {target_masks.shape}')
    if images.size(0) != target_params.size(0):
        print('\nNo. images does not match no. targets!! Exiting.\n')
        sys.exit(0)

    dataset = load_data.ResonanceDataset(images, target_params, target_masks, gradients=gradients)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = random_split(dataset, \
                                     [train_size, val_size], \
                                        generator=torch.Generator().manual_seed(SEED))
    
    print(f'Training size: {len(train_dataset)}')
    print(f'Validation size: {len(val_dataset)}')
    
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers)
    
    val_loader = DataLoader(val_dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=num_workers)

    net = MultiRes_NumLevels_CNN(in_ch=2,
                                    base=base,
                                    dropout_p=dropout_p,
                                    kernel_size=kernel_size,
                                    max_levels=load_data.MAX_RESONANCES).to(device)
    print('Loaded multi. resonance num levels CNN')

    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )

    results = {
        "epoch": [],
        "train_loss": [], "train_mae": [], "train_acc_exact": [], "train_acc_within_1": [],
        "val_loss": [], "val_mae": [], "val_acc_exact": [], "val_acc_within_1": []
    }

    print("\n------------------------------------\n")

    for epoch in range(1, num_epochs + 1):

        train_m = train_epoch(net, train_loader, optimizer, device, grad_clip=1.0)
        val_m = eval_epoch(net, val_loader, device)

        scheduler.step(val_m['mae'])

        results["epoch"].append(epoch)
        results["train_loss"].append(train_m["loss"])
        results["train_mae"].append(train_m["mae"])
        results["train_acc_exact"].append(train_m["acc_exact"])
        results["train_acc_within_1"].append(train_m["acc_within_1"])
        
        results["val_loss"].append(val_m["loss"])
        results["val_mae"].append(val_m["mae"])
        results["val_acc_exact"].append(val_m["acc_exact"])
        results["val_acc_within_1"].append(val_m["acc_within_1"])

        if epoch % 5 == 0:
            print(
                f"Epoch {epoch:02d} |"
                f"\n\tTrain loss {train_m['loss']:.4f}, MAE {train_m['mae']:.3f}, "
                f"acc {train_m['acc_exact']:.3f}, acc ±1 {train_m['acc_within_1']:.3f}"
                f"\n\tVal loss {val_m['loss']:.4f}, MAE {val_m['mae']:.3f}, "
                f"acc {val_m['acc_exact']:.3f}, acc ±1 {val_m['acc_within_1']:.3f}\n"
            )

if __name__ == "__main__":

    # prompt user for parameters
    # training_path = input("Training data path: ")
    # images_path = input("Input images path: ")
    # subset_size = int(input("Subset size: "))
    # cropping_strength = float(input("Cropping strength: "))
    # num_epochs = int(input("Num. epochs: "))
    # batch_size = int(input("Batch size: "))
    # num_workers = int(input("Num. workers: "))

    print("\n------------------------------------\n")

    main()