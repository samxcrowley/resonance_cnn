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

num_workers = 32
subset_size = 1000

cropping_strength = sys.argv[1]

# training
num_epochs = 100
batch_size = 32
lr = 1e-3
weight_decay = 1e-4

# model params.
dropout_p = 0.3
base = 80
kernel_size = 3
gradients = True

def train_epoch(net, loader, optimizer, device, grad_clip=None):

    net.train()

    running = {
        'loss': 0.0, 'mae': 0.0, 'count': 0
    }

    for batch_images, batch_targets in loader:

        batch_images = batch_images.to(device=device, dtype=torch.float32)
        batch_targets = batch_targets.to(device=device, dtype=torch.float32)

        E_target = batch_targets

        E_pred = net(batch_images)

        loss = F.mse_loss(E_pred, E_target)

        optimizer.zero_grad(set_to_none=True)

        loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(net.parameters(), grad_clip)
        optimizer.step()

        with torch.no_grad():
            running['loss'] += loss.item() * batch_images.size(0)
            running['mae'] += torch.abs(E_pred - E_target).sum().item()
            running['count'] += batch_images.size(0)

    n = running['count']

    return {
        'loss': running['loss'] / n,
        'mae': running['mae'] / n
    }
    
def eval_epoch(net, loader, device):

    net.eval()
    
    running = {
        'loss': 0.0, 'mae': 0.0, 'count': 0
    }

    with torch.no_grad():

        for batch_images, batch_targets in loader:

            batch_images = batch_images.to(device=device, dtype=torch.float32)
            batch_targets = batch_targets.to(device=device, dtype=torch.float32)

            E_target = batch_targets

            E_pred = net(batch_images)

            loss = F.mse_loss(E_pred, E_target)

            running['loss'] += loss.item() * batch_images.size(0)
            running['mae'] += torch.abs(E_pred - E_target).sum().item()
            running['count'] += batch_images.size(0)

    n = running['count']
    metrics = {
        'loss': running['loss'] / n,
        'mae': running['mae'] / n
    }

    return metrics
    
def main():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    images, targets = \
        load_data.load_singleres_images_and_targets(cropping_strength)

    if cropping_strength == 0.0:
        targets = targets[::10]

    # if we have defined a smaller subset, cut off the unneeded samples
    if subset_size < len(images):
        images = images[:subset_size]
        targets = targets[:subset_size]

    print(f'Images shape: {images.shape}')
    print(f'Targets shape: {targets.shape}')
    if images.size(0) != targets.size(0):
        print('\nNo. images does not match no. targets!! Exiting.\n')
        sys.exit(0)
    print(f'Cropping strength: {cropping_strength}')

    dataset = load_data.SingleResonanceDataset(images, targets, gradients=gradients)

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
    print("\n------------------------------------\n")

    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    results = {
        "epoch": [],
        "train_loss": [], "train_mae": [],
        "val_loss": [], "val_mae": []
    }

    # track best model
    best_val_loss = float('inf')
    best_epoch = 0
    best_model_state = None

    for epoch in range(1, num_epochs + 1):

        train_m = train_epoch(net, train_loader, optimizer, device)
        val_m = eval_epoch(net, val_loader, device)

        scheduler.step(val_m['loss'])

        results["epoch"].append(epoch)
        results["train_loss"].append(train_m["loss"])
        results["train_mae"].append(train_m["mae"])

        results["val_loss"].append(val_m["loss"])
        results["val_mae"].append(val_m["mae"])

        # track best model
        if val_m['loss'] < best_val_loss:
            best_val_loss = val_m['loss']
            best_epoch = epoch
            best_model_state = {k: v.cpu().clone() for k, v in net.state_dict().items()}

        if epoch % 5 == 0:
            print(
                f"Epoch {epoch:02d} | "
                f"train loss {train_m['loss']:.4f}"
                f" | val loss {val_m['loss']:.4f}"
                f" | train MAE {train_m['mae']:.4f}"
                f" | val MAE {val_m['mae']:.4f}"
            )

    # save best model
    model_filename = \
        f'results/singleres/energy/{cropping_strength}crop_model.pt'
    os.makedirs(os.path.dirname(model_filename), exist_ok=True)
    torch.save(best_model_state, model_filename)
    print(f'Model saved to {model_filename}')

    # save results data
    results_filename = \
        f'results/singleres/energy/{cropping_strength}crop_results.csv'
    os.makedirs(os.path.dirname(results_filename), exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(results_filename, index=False)
    print(f'Results saved to {results_filename}')

if __name__ == "__main__":
    main()