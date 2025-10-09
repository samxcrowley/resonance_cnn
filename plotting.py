import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import preprocessing
import load_data

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

# plot train and val loss of one training run
def plot_results(subdir, cropping_strength):

    df = pd.read_csv(f'results/{subdir}/{cropping_strength}crop_results.csv')

    plt.figure()
    plt.plot(df["epoch"], df["train_loss_E"], label="train loss")
    plt.plot(df["epoch"], df["val_loss_E"], label="val loss")
    # plt.plot(df["epoch"], df["train_mae_E"], label="MAE")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Total loss")
    plt.legend()
    plt.tight_layout()

    filename = f'plots/{subdir}/{cropping_strength}crop_results.png'
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename, dpi=150)

# plot train and val loss of all cropping strengths
def plot_losses(subdir, strengths=[0.0, 0.5, 0.75, 0.9]):

    files = []

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax1, ax2 = axes

    for _s in strengths:

        name = f'results/{subdir}/{_s}crop_results.csv'

        df = pd.read_csv(name)

        x = df["epoch"].values
        xlabel = "Epoch"

        # Y columns
        y1 = df["train_loss_E"].values
        y2 = df["val_loss_E"].values

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
    plt.savefig(f"plots/{subdir}/allcrop_results.png")

###
# multi-res plotting
###

def get_all_predictions(model, dataloader, device):

    model.eval()
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():

        for batch_images, batch_target_params, batch_target_masks in dataloader:

            batch_images = batch_images.to(device=device, dtype=torch.float32)
            batch_target_masks = batch_target_masks.to(device=device, dtype=torch.bool)
            
            num_target = batch_target_masks.sum(dim=1).cpu().numpy()
            num_pred = model(batch_images).cpu().numpy()
            
            all_preds.extend(num_pred)
            all_targets.extend(num_target)
    
    predictions = np.array(all_preds)
    predictions_rounded = np.round(predictions).astype(int)
    targets = np.array(all_targets, dtype=int)
    
    return predictions, predictions_rounded, targets

def plot_confusion_matrix(targets, predictions_rounded,
                          save_path='plots/multires/confusion_matrix.png'):
    
    data_max = max(targets.max(), predictions_rounded.max())
    axis_max = int(np.ceil(data_max / 2) * 2) # round to multiple of 2

    # unique labels
    labels = list(range(0, axis_max + 1))
    
    cm = confusion_matrix(targets, predictions_rounded, labels=labels)
    
    fig, ax = plt.subplots(figsize=(18, 7))
    title = 'Num. Resonances Confusion Matrix'
    
    disp1 = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp1.plot(ax=ax, cmap='Blues', values_format='d', colorbar=True)
    ax.set_title(f'{title}', fontsize=14, fontweight='bold')
    ax.set_xlabel('Predicted Num. Resonances', fontsize=12)
    ax.set_ylabel('True Num. Resonances', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f'Confusion matrix saved to {save_path}')
    plt.close()

def plot_prediction_scatter(targets, predictions, predictions_rounded, 
                            save_path='plots/multires/prediction_scatter.png'):

    fig, ax = plt.subplots(figsize=(16, 7))

    data_max = max(targets.max(), predictions.max())
    axis_max = int(np.ceil(data_max / 2) * 2) # round to nearest multiple of 2
    
    ax.scatter(targets, predictions, alpha=0.5, s=40, edgecolors='black', linewidth=0.5)

    # line
    min_val = min(targets.min(), predictions.min())
    max_val = max(targets.max(), predictions.max())
    ax.plot([min_val, max_val], [min_val, max_val], 
                 'r--', linewidth=2.5, label='Perfect prediction', zorder=5)
    
    # mean prediction per class
    unique_targets = sorted(set(targets))
    for true_count in unique_targets:
        mask = targets == true_count
        mean_pred = np.mean(predictions[mask])
        ax.scatter([true_count], [mean_pred], 
                       color='red', s=150, marker='D', 
                       edgecolors='black', linewidth=2, zorder=6, alpha=0.8)
    
    ax.set_xlabel('True Number of Resonances', fontsize=13, fontweight='bold')
    ax.set_ylabel('Predicted Number of Resonances', fontsize=13, fontweight='bold')
    ax.set_title('Predicted vs. True Num. Resonances', fontsize=14, fontweight='bold')

    ax.set_xlim(-0.5, axis_max + 0.5)
    ax.set_ylim(-0.5, axis_max + 0.5)
    
    # ticks every 2 units
    ax.set_xticks(range(0, axis_max + 1, 2))
    ax.set_yticks(range(0, axis_max + 1, 2))

    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f'Prediction scatter plot saved to {save_path}')
    plt.close()