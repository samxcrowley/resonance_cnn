import os
import sys
import torch
from torch.utils.data import DataLoader, random_split
import gzip
import json
import plotting
import load_data
import preprocessing
from multires_numlevels_cnn import MultiRes_NumLevels_CNN

SEED = 22

cropping_strength = sys.argv[1]
num_res = sys.argv[2]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

net = MultiRes_NumLevels_CNN(in_ch=2,
                                    base=80,
                                    dropout_p=0.3,
                                    kernel_size=3,
                                    max_levels=load_data.MAX_RESONANCES).to(device)
checkpoint = torch.load(f'results/{num_res}res/{cropping_strength}crop_model.pt')
net.load_state_dict(checkpoint)

images, target_params, target_masks = \
   load_data.load_images_and_targets(num_res, cropping_strength)
dataset = load_data.ResonanceDataset(images, target_params, target_masks, gradients=True)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
    
train_dataset, val_dataset = random_split(dataset, \
                                     [train_size, val_size], \
                                        generator=torch.Generator().manual_seed(SEED))

val_loader = DataLoader(val_dataset,
                            batch_size=32,
                            shuffle=False,
                            num_workers=32)

preds, preds_rounded, targets = plotting.get_all_predictions(net, val_loader, device)

dir_ = f'plots/{num_res}res/{cropping_strength}'
os.makedirs(dir_, exist_ok=True)

plotting.plot_confusion_matrix(targets, preds_rounded, \
         save_path=f'{dir_}/confusion_matrix.png')

plotting.plot_prediction_scatter(targets, preds, preds_rounded, \
         save_path=f'{dir_}/scatter.png')