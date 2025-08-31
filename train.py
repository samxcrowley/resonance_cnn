import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split, Subset
import matplotlib.pyplot as plt
import utils
import model
import masked_model
import data_loading
import sys
import math

SEED = 22

n_epochs = 100
batch_size = 16
lr = 1e-4
weight_decay = 1e-4

tiny_size = 48
tiny_train_size = 32

dropout_p = 0.0
in_ch = 2
base = 80
kernel_size = 3
gradients = False

def train_epoch(net, loader, optimizer, device, grad_clip=None):

    net.train()

    running = {
        'loss': 0.0, 'loss_E': 0.0, 'loss_gamma': 0.0,
        'mae_E': 0.0, 'mae_gamma': 0.0, 'count': 0
    }

    for batch_images, batch_targets in loader:

        batch_images = batch_images.to(device=device, dtype=torch.float32)
        batch_targets = batch_targets.to(device=device, dtype=torch.float32)

        E_target = batch_targets[:, 0]
        gamma_target = batch_targets[:, 1]

        optimizer.zero_grad()

        E_pred, gamma_pred = net(batch_images)
        loss_E = F.mse_loss(E_pred, E_target)
        loss_gamma = F.mse_loss(gamma_pred, gamma_target)
        loss = loss_E + loss_gamma

        loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(net.parameters(), grad_clip)
        optimizer.step()

        with torch.no_grad():
            running['loss'] += loss.item() * batch_images.size(0)
            running['loss_E'] += loss_E.item() * batch_images.size(0)
            running['loss_gamma'] += loss_gamma.item() * batch_images.size(0)
            running['mae_E'] += torch.mean(torch.abs(E_pred - E_target)).item() * batch_images.size(0)
            running['mae_gamma'] += torch.mean(torch.abs(gamma_pred - gamma_target)).item() * batch_images.size(0)
            running['count'] += batch_images.size(0)

        n = running['count']

        return {
            'loss': running['loss'] / n,
            'loss_E': running['loss_E'] / n,
            'loss_gamma': running['loss_gamma'] / n,
            'mae_E': running['mae_E'] / n,
            'mae_gamma': running['mae_gamma'] / n,
        }
    
def evaluate(net, loader, device):

    net.eval()
    
    running = {
        'loss': 0.0, 'loss_E': 0.0, 'loss_gamma': 0.0,
        'mae_E': 0.0, 'mae_gamma': 0.0, 'count': 0
    }

    all_gamma_pred, all_gamma_target = [], []

    for batch_images, batch_targets in loader:

        batch_images = batch_images.to(device=device, dtype=torch.float32)
        batch_targets = batch_targets.to(device=device, dtype=torch.float32)

        E_target = batch_targets[:, 0]
        gamma_target = batch_targets[:, 1]

        E_pred, gamma_pred = net(batch_images)

        loss_E = F.mse_loss(E_pred, E_target)
        loss_gamma = F.mse_loss(gamma_pred, gamma_target)
        loss = loss_E + loss_gamma

        running['loss'] += loss.item() * batch_images.size(0)
        running['loss_E'] += loss_E.item() * batch_images.size(0)
        running['loss_gamma'] += loss_gamma.item() * batch_images.size(0)
        running['mae_E'] += torch.mean(torch.abs(E_pred - E_target)).item() * batch_images.size(0)
        running['mae_gamma']  += torch.mean(torch.abs(gamma_pred - gamma_target)).item() * batch_images.size(0)
        running['count'] += batch_images.size(0)

        all_gamma_pred.append(gamma_pred.cpu())
        all_gamma_target.append(gamma_target.cpu())

    n = running['count']
    metrics = {
        'loss': running['loss']   / n,
        'loss_E': running['loss_E'] / n,
        'loss_gamma': running['loss_gamma'] / n,
        'mae_E': running['mae_E']  / n,
        'mae_gamma': running['mae_gamma']  / n,
    }

    # R^2 for gamma
    if len(all_gamma_pred) > 0:
        y = torch.cat(all_gamma_target).detach().cpu().numpy()
        yhat = torch.cat(all_gamma_pred).detach().cpu().numpy()
        ss_res = float(np.sum((y - yhat)**2))
        ss_tot = float(np.sum((y - np.mean(y))**2) + 1e-12)
        metrics['r2_gamma'] = 1.0 - ss_res / ss_tot

    return metrics
    
def main():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    path = 'data/o16/o16_training_new.gz'
    
    # images = data_loading.get_images(path, log=True, crop_coef=2.5, angle_p=0.2)
    images = torch.load('images.pt')
    print(images.shape)

    targets = data_loading.get_targets(path)

    dataset = data_loading.ResonanceDataset(images, targets, gradients=gradients)

    # train_size = int(0.8 * len(dataset))
    # val_size = len(dataset) - train_size
    
    subset = Subset(dataset, list(range(tiny_size)))

    train_dataset, val_dataset = random_split(subset, \
                                              [tiny_train_size, tiny_size - tiny_train_size], \
                                                generator=torch.Generator().manual_seed(SEED))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    net = masked_model.ResonanceCNN_Masked(in_ch=in_ch, base=base, dropout_p=dropout_p, kernel_size=kernel_size).to(device)

    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    best_val = math.inf
    best_state = None

    history = {
        "epoch": [],
        "train_loss": [], "train_loss_E": [], "train_loss_G": [], "train_mae_E": [], "train_mae_G": [],
        "val_loss": [],   "val_loss_E": [],   "val_loss_G": [],   "val_mae_E": [],   "val_mae_G": []
    }

    for epoch in range(1, n_epochs + 1):

        train_m = train_epoch(net, train_loader, optimizer, device, grad_clip=1.0)
        val_m = evaluate(net, val_loader, device)

        # scheduler.step(val_m['loss'])

        history["epoch"].append(epoch)
        history["train_loss"].append(train_m["loss"])
        history["train_loss_E"].append(train_m["loss_E"])
        history["train_loss_G"].append(train_m["loss_gamma"])
        history["train_mae_E"].append(train_m["mae_E"])
        history["train_mae_G"].append(train_m["mae_gamma"])

        history["val_loss"].append(val_m["loss"])
        history["val_loss_E"].append(val_m["loss_E"])
        history["val_loss_G"].append(val_m["loss_gamma"])
        history["val_mae_E"].append(val_m["mae_E"])
        history["val_mae_G"].append(val_m["mae_gamma"])

        if epoch % 5 == 0:
            print(
                f"Epoch {epoch:02d} | "
                f"train loss {train_m['loss']:.4f} (E {train_m['loss_E']:.4f}, G {train_m['loss_gamma']:.4f}) "
                f"| val loss {val_m['loss']:.4f} (E {val_m['loss_E']:.4f}, G {val_m['loss_gamma']:.4f}) "
                f"| val MAE(E) {val_m['mae_E']:.4f} MAE(G) {val_m['mae_gamma']:.4f} "
                f"{'| r2(logG) %.3f' % val_m['r2_logG'] if 'r2_logG' in val_m else ''}"
            )

        if val_m['loss'] < best_val:
            best_val = val_m['loss']
            best_state = {k: v.cpu() for k, v in net.state_dict().items()}

    if best_state is not None:
        net.load_state_dict(best_state)
        torch.save(net.state_dict(), 'resonance_cnn_best.pt')
        print(f"Saved best model with val loss {best_val:.4f} -> resonance_cnn_best.pt")

    hist_df = pd.DataFrame(history)
    hist_df.to_csv("training_history.csv", index=False)
    plt.figure()
    plt.plot(hist_df["epoch"], hist_df["train_loss"], label="train loss")
    plt.plot(hist_df["epoch"], hist_df["val_loss"],   label="val loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Total loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig("loss_total.png", dpi=150)

if __name__ == "__main__":

    main()