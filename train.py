import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split, Subset
import matplotlib.pyplot as plt
import utils
import model
import partial_model
import data_loading
import sys
import math

SEED = 22

path = 'data/o16/o16_training_new.gz'

# training
n_epochs = 100
batch_size = 16
lr = 1e-4
weight_decay = 1e-4

# n. samples
subset_size = 64 # n. total
subset_train_size = 48 # n. training samples

# model params.
using_partial_model = True
dropout_p = 0.0
in_ch = 1
base = 80
kernel_size = 3
gradients = True

# image cropping
crop_coef = 2.0
angle_p = 0.5

def train_epoch(net, loader, optimizer, device, grad_clip=None):

    net.train()

    running = {
        'loss': 0.0, 'loss_E': 0.0,
        'mae_E': 0.0, 'count': 0
    }

    for batch_images, batch_targets in loader:

        batch_images = batch_images.to(device=device, dtype=torch.float32)
        batch_targets = batch_targets.to(device=device, dtype=torch.float32)

        E_target = batch_targets[:, 0]

        E_pred = net(batch_images)
        loss_E = F.mse_loss(E_pred, E_target)
        loss = loss_E

        optimizer.zero_grad(set_to_none=True)

        loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(net.parameters(), grad_clip)
        optimizer.step()

        with torch.no_grad():
            running['loss'] += loss.item() * batch_images.size(0)
            running['loss_E'] += loss_E.item() * batch_images.size(0)
            running['mae_E'] += torch.mean(torch.abs(E_pred - E_target)).item() * batch_images.size(0)
            running['count'] += batch_images.size(0)

    n = running['count']

    return {
        'loss': running['loss'] / n,
        'loss_E': running['loss_E'] / n,
        'mae_E': running['mae_E'] / n
    }
    
def eval_epoch(net, loader, device):

    net.eval()
    
    running = {
        'loss': 0.0, 'loss_E': 0.0,
        'mae_E': 0.0, 'count': 0
    }

    with torch.no_grad():
        for batch_images, batch_targets in loader:

            batch_images = batch_images.to(device=device, dtype=torch.float32)
            batch_targets = batch_targets.to(device=device, dtype=torch.float32)

            E_target = batch_targets[:, 0]

            E_pred = net(batch_images)

            loss_E = F.mse_loss(E_pred, E_target)
            loss = loss_E

            running['loss'] += loss.item() * batch_images.size(0)
            running['loss_E'] += loss_E.item() * batch_images.size(0)
            running['mae_E'] += torch.mean(torch.abs(E_pred - E_target)).item() * batch_images.size(0)
            running['count'] += batch_images.size(0)

    n = running['count']
    metrics = {
        'loss': running['loss']   / n,
        'loss_E': running['loss_E'] / n,
        'mae_E': running['mae_E']  / n
    }

    return metrics
    
def main():

    # create parameters strings for file saving
    images_params_str = f'crop_{crop_coef}_angle_{angle_p}'
    params_str = f'samples_{subset_train_size}_crop_{crop_coef}_angle_{angle_p}_epochs_{n_epochs}'
    if using_partial_model:
        params_str = 'partial_' + params_str
    else:
        params_str = 'cnn_' + params_str

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    images_path = f'data/images/{images_params_str}.pt'

    images = data_loading.get_images(path, crop_coef=crop_coef, angle_p={angle_p})
    torch.save(images, images_path)
    # images = torch.load(images_path)

    print(f'Images at {images_path}')

    # only the partial CNN needs the mask channel
    if not using_partial_model:
        images = images[:, 0:1, :, :]

    # targets only get duplicated (by data_loading.IMG_DUP) when images are cropped
    if crop_coef == 0:
        targets = data_loading.get_targets(path, dup=False)
    else:
        targets = data_loading.get_targets(path, dup=True)

    print(f'Images shape: f{images.shape}')
    print(f'Targets shape: f{targets.shape}')
    if images.size(0) != targets.size(0):
        print('\nWarning: no. images does not match no. targets!! Exiting.\n')
        sys.exit(0)

    dataset = data_loading.ResonanceDataset(images, targets, gradients=gradients)

    # train_size = int(0.8 * len(dataset))
    # val_size = len(dataset) - train_size
    
    subset = Subset(dataset, list(range(subset_size)))
    train_dataset, val_dataset = random_split(subset, \
                                              [subset_train_size, subset_size - subset_train_size], \
                                                generator=torch.Generator().manual_seed(SEED))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    if using_partial_model:
        net = using_partial_model.ResonanceCNN_Masked(in_ch=in_ch, base=base, dropout_p=dropout_p, kernel_size=kernel_size).to(device)
    else:
        net = model.ResonanceCNN(in_ch=in_ch, base=base, dropout_p=dropout_p, kernel_size=kernel_size).to(device)

    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)

    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    history = {
        "epoch": [],
        "train_loss": [], "train_loss_E": [], "train_mae_E": [],
        "val_loss": [], "val_loss_E": [], "val_mae_E": []
    }

    for epoch in range(1, n_epochs + 1):

        train_m = train_epoch(net, train_loader, optimizer, device, grad_clip=1.0)
        val_m = eval_epoch(net, val_loader, device)

        # scheduler.step(val_m['loss'])

        history["epoch"].append(epoch)
        history["train_loss"].append(train_m["loss"])
        history["train_loss_E"].append(train_m["loss_E"])
        history["train_mae_E"].append(train_m["mae_E"])

        history["val_loss"].append(val_m["loss"])
        history["val_loss_E"].append(val_m["loss_E"])
        history["val_mae_E"].append(val_m["mae_E"])

        if epoch % 5 == 0:
            print(
                f"Epoch {epoch:02d} | "
                f"train loss {train_m['loss']:.4f}"
                f" | val loss {val_m['loss']:.4f}"
                f" | train MAE {train_m['mae_E']:.4f}"
                f" | val MAE {val_m['mae_E']:.4f}"
            )

    df = pd.DataFrame(history)
    df.to_csv(f'data/training_data/{params_str}.csv', index=False)

if __name__ == "__main__":
    main()