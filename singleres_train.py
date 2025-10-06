import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import load_data
from singleres_energylevel_cnn import SingleRes_EnergyLevel_CNN
import preprocessing
import sys
import os

SEED = 22

training_path = f'data/o16/o16_training_small.gz'
images_path = f'data/images.pt'

num_workers = 32
subset_size = 1000

cropping_strength = 0.0

# training
num_epochs = 100
batch_size = 32
lr = 1e-4
weight_decay = 1e-4

# model params.
dropout_p = 0.0
base = 80
kernel_size = 3
gradients = False

def train_epoch(net, loader, optimizer, device, grad_clip=None):

    net.train()

    running = {
        'loss': 0.0, 'loss_E': 0.0,
        'mae_E': 0.0, 'count': 0
    }

    for batch_images, batch_target_params, batch_target_masks in loader:

        batch_images = batch_images.to(device=device, dtype=torch.float32)
        batch_target_params = batch_target_params.to(device=device, dtype=torch.float32)
        batch_target_masks = batch_target_masks.to(device=device, dtype=torch.bool)

        E_target = batch_target_params[:, 0, 0]

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
            running['mae_E'] += torch.abs(E_pred - E_target).sum().item()
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

        for batch_images, batch_target_params, batch_target_masks in loader:

            batch_images = batch_images.to(device=device, dtype=torch.float32)
            batch_target_params = batch_target_params.to(device=device, dtype=torch.float32)
            batch_target_masks = batch_target_masks.to(device=device, dtype=torch.bool)

            E_target = batch_target_params[:, 0, 0]

            E_pred = net(batch_images)

            loss_E = F.mse_loss(E_pred, E_target)
            loss = loss_E

            running['loss'] += loss.item() * batch_images.size(0)
            running['loss_E'] += loss_E.item() * batch_images.size(0)
            running['mae_E'] += torch.abs(E_pred - E_target).sum().item()
            running['count'] += batch_images.size(0)

    n = running['count']
    metrics = {
        'loss': running['loss']   / n,
        'loss_E': running['loss_E'] / n,
        'mae_E': running['mae_E']  / n
    }

    return metrics
    
def main():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    images = torch.load('data/images/singleres_images_crop0.0.pt')
    target_params = torch.load('data/targets/singleres_targetparams_crop0.0.pt')
    target_masks = torch.load('data/targets/singleres_targetmasks_crop0.0.pt')

    images = images[::10]
    target_params = target_params[::10]
    target_masks = target_masks[::10]

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
                            shuffle=True,
                            num_workers=num_workers)

    net = SingleRes_EnergyLevel_CNN(in_ch=2,
                                    base=base,
                                    dropout_p=dropout_p,
                                    kernel_size=kernel_size).to(device)
    print('Loaded single resonance energy level CNN')

    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    results = {
        "epoch": [],
        "train_loss": [], "train_loss_E": [], "train_mae_E": [],
        "val_loss": [], "val_loss_E": [], "val_mae_E": []
    }

    for epoch in range(1, num_epochs + 1):

        train_m = train_epoch(net, train_loader, optimizer, device)
        val_m = eval_epoch(net, val_loader, device)

        scheduler.step(val_m['loss'])

        results["epoch"].append(epoch)
        results["train_loss"].append(train_m["loss"])
        results["train_loss_E"].append(train_m["loss_E"])
        results["train_mae_E"].append(train_m["mae_E"])

        results["val_loss"].append(val_m["loss"])
        results["val_loss_E"].append(val_m["loss_E"])
        results["val_mae_E"].append(val_m["mae_E"])

        if epoch % 5 == 0:
            print(
                f"Epoch {epoch:02d} | "
                f"train loss {train_m['loss']:.4f}"
                f" | val loss {val_m['loss']:.4f}"
                f" | train MAE {train_m['mae_E']:.4f}"
                f" | val MAE {val_m['mae_E']:.4f}"
            )

    # save results data
    results_filename = \
        f'results/{cropping_strength}crop_{subset_size}subset_{num_epochs}epochs_{batch_size}batch.csv'

    os.makedirs(os.path.dirname(results_filename), exist_ok=True)

    df = pd.DataFrame(results)
    df.to_csv(results_filename, index=False)

    print(f'Results saved to {results_filename}')

if __name__ == "__main__":

    # prompt user for parameters
    # training_path = input("Training data path: ") or training_path
    # images_path = input("Input images path: ") or images_path
    # subset_size = int(input("Subset size: "))
    # cropping_strength = float(input("Cropping strength: "))
    # num_epochs = int(input("Num. epochs: "))
    # batch_size = int(input("Batch size: "))
    # num_workers = int(input("Num. workers: "))

    print("\n------------------------------------\n")

    main()