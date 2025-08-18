import torch
from torch.utils.data import DataLoader, Dataset, random_split
import utils
import model
import data_loading

if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    path = 'data/o16/o16_training.gz'

    images = data_loading.get_images(path, log=True) # shape: [1000, 4, 101, 7]
    targets = data_loading.get_targets(path) # shape: [1000, 2]

    dataset = data_loading.ResonanceDataset(images, targets)
    
    train_size = int(0.8 * len(dataset)) # 80/20 training/validation split
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])