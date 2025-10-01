import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import preprocessing
import os

def plot_image(image, name):

    E_axis, A_axis = preprocessing.global_grid()

    values = image[0].numpy() if isinstance(image, torch.Tensor) else image[0]
    mask = image[1].numpy() if isinstance(image, torch.Tensor) else image[1]

    # mask out invalid values
    plot_data = np.where(mask == 1, values, np.nan)

    plt.figure(figsize=(6, 4))
    cmap = plt.cm.viridis.copy()
    cmap.set_bad(color='black')

    plt.pcolormesh(A_axis, E_axis, plot_data, cmap=cmap, shading='auto')
    plt.colorbar(label="dsdO")
    plt.xlabel("Angle")
    plt.ylabel("Energy")

    filename = f'plots/images/{name}.png'
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename, dpi=150)

def plot_results(results_name):

    df = pd.read_csv(f'results/{results_name}.csv')

    plt.figure()
    plt.plot(df["epoch"], df["train_loss"], label="train loss")
    plt.plot(df["epoch"], df["val_loss"], label="val loss")
    plt.plot(df["epoch"], df["train_mae_E"], label="MAE")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Total loss")
    plt.legend()
    plt.tight_layout()

    filename = f'plots/results/{results_name}.png'
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename, dpi=150)

def plot_losses():

    s = [0.0, 0.5, 0.75, 0.9]
    files = []

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax1, ax2 = axes

    for _s in s:

        name = f'down_results/groupnorm/True-partial_{_s}crop_2000subset_100epochs_32batch.csv'

        df = pd.read_csv(name)

        x = df["epoch"].values
        xlabel = "Epoch"

        # Y columns
        y1 = df["train_loss"].values
        y2 = df["val_loss"].values

        # Plot both lines
        ax1.plot(x, y1, label=_s)
        ax2.plot(x, y2, label=_s)

    ax1.set_xlabel(xlabel)
    ax1.set_ylabel("Train Loss")
    ax2.set_xlabel(xlabel)
    ax2.set_ylabel("Val. loss")

    ax1.legend(title="Strength")
    ax2.legend(title="Strength")

    for ax in (ax1, ax2):
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("plots/results/groupnorm_2000subset_trainandval.png")