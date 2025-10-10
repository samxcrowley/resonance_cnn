import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import load_data, plotting, preprocessing
from multires_numlevels_cnn import MultiRes_NumLevels_CNN
from singleres_energylevel_cnn import SingleRes_EnergyLevel_CNN

# image range is [9.118, 12.126]
# resonances at 9.586, 10.3581, 11.5055, 11.5058
exp_min = 9.12
exp_max = 12.13

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'

E_axis, A_axis = preprocessing.global_grid()
E_min = E_axis.min()
E_max = E_axis.max()

# load base image
exp_img = load_data.get_exp_image('data/o16/exp.gz', log_cx=True)
exp_img = preprocessing.sobel(exp_img)
plotting.plot_image(exp_img, 'exp_img')

df = pd.DataFrame(columns=['centre_energy', 'predicted_resonance_energy'])

# load model
net = SingleRes_EnergyLevel_CNN(in_ch=2,
                                    base=80,
                                    dropout_p=0.3,
                                    kernel_size=3)
checkpoint = torch.load(f'results/singleres/energy/0.25crop_model.pt')
net.load_state_dict(checkpoint)
net.eval()

window_width = 0.25
start = exp_min + (window_width / 2)
end = exp_max - (window_width / 2)
step = 0.1
for E in np.arange(start, end, step):

    sub_img = load_data.get_subset_of_image(exp_img, E, E_axis, window_width)
    sub_img = sub_img.unsqueeze(0).to(device=device, dtype=torch.float32)

    with torch.no_grad():

        unnorm_E = net(sub_img).item()

        pred_E = unnorm_E * (E_max - E_min) + E_min

        df.loc[len(df)] = [E, pred_E]

df.to_csv('results/sliding_window.csv', index=False)
# df = pd.read_csv('results/sliding_window.csv')

plt.figure(figsize=(6, 4))
plt.plot(df['centre_energy'], df['predicted_resonance_energy'], marker='o', linestyle='-', color='tab:blue', label='data')
plt.title(f'Sliding window (width {window_width})')
plt.xlabel('Window centre energy')
plt.ylabel('Predicted energy')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('plots/sliding_window.png')