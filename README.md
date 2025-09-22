# Overview

A CNN to identify and predict resonances and their properties from nuclear scattering cross-sections. For now, the model is only trained on $^{12}\text{C} + \alpha \to {}^{12}\text{C} + \alpha$ scattering data.

# Usage

- TODO: Script to generate data (turn `.gz` training data into `.pt` images data, placed on the grid and cropped)
- Run the model by calling `python3 train.py`. This will prompt you for the training data path, images data path, subset size (how many samples to train on), batch size, number of epochs, and number of workers (for multiple CPU cores)
- Results data, including train/val loss and train/val MAE over all epochs, are saved as .csv files in `results/`
- Plot results data with `utils.plot_results(filename)`. Plots are saved to `plots/results/`

# Input

Here, nuclear cross-section data are treated and referred to as images as they can very sensibly be visualised as such, and our machine learning techniques are borrowed from image classification and feature detection.

## Training data

Raw data comes as `json` files compressed into `.gz` files. They store the cross-section and target data for a number of samples.

## Images

- Images are placed on a global data grid, which has axes defined in `data_loading.py`
- Each image has 2 channels; the cross-section value and a visibility mask
- Where there is no image data at a coordinate, that coordinate's mask is set to `0`. Otherwise, the mask is `1`
- Images are stored in a tensor of size `[n_samples, 2, n_E, n_A]`, where `n_E` and `n_A` are the numbers of energy and angle bins in the global data grid

### Image cropping

To simulate patchy and irregular experimental data, the input images can be cropped with `utils.crop_image()`. This is done by changing ones to zeroes in the visibility mask channel, and not by altering the cross-section channel.

Random energy and angle ranges are chosen, and each raw image from the dataset is duplicated `data_loading.IMG_DUP` times (default is 10) times, with each being cropped randomly.

## Targets

Target data, for now, includes energy and total width (gamma) in a tensor of size `[n_samples, 2]`.
- Energy of n-th sample: `[n, 0]`
- Gamma of n-th sample: `[n, 1]`

# Model

There are two versions of the model:
- ResonanceCNN (`model.py`) -- A normal CNN run on 1-channel image data
- ResonancePartialCNN (`partial_model.py`) -- A partial CNN run on 2-channel image data, where channel 2 is a visibility mask
    - Convolutions are weighted by the proportion of visible parameters in a sliding window
    - Windows with a small amount of visible parameters are weighted higher, i.e. if `K` is the number of parameters in a window, and `valid_count` is the number of those parameters that match to a `1` in the mask channel, then the convolution is scaled by `K / valid_count`
    - Non-zero parameters that match to a `0` in the mask channel are counted as `0`

By default, the Partial CNN is used, however this can be changed by setting the `using_partial_model` flag in `train.py` to False.

# Output

For now, the model only predicts the energy level of detected resonances. Soon (hopefully) it will be extended to predict total width, number of resonances, spin, parity, partial widths ...