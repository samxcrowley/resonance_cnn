# Input

Here, nuclear cross-section data are treated and referred to as images as they can very sensibly be visualised as such, and our machine learning techniques are borrowed from image classification and feature detection.

## Images

Image data are saved to `data/images`.

- Images are placed on a global data grid, which has axes defined in `data_loading.py`
- Each image has 2 channels; the cross-section value and a visibility mask
- Where there is no image data at a coordinate, that coordinate's mask is set to `0`. Otherwise, the mask is `1`
- Images are stored in a tensor of size `[n_samples, 2, n_E, n_A]`, where `n_E` and `n_A` are the numbers of energy and angle bins in the global data grid

### Image cropping

To simulate patchy and irregular experimental data, the input images can be cropped according to parameters `crop_coef` and `angle_p`. This is done by changing ones to zeroes in the visibility mask channel, and not by altering the cross-section channel.

- Energy ranges at each angle are cropped from the bottom and top, each by up to `1 / crop_coef`
- Angles are dropped from the images at random, by a percentage of up to `angle_p`
- If `crop_coef = 0` the images are left unchanged
- If `crop_coef > 0` the images are each duplicated `data_loading.IMG_DUP` times (default is 10) and each of these are cropped at random, and the targets are duplicated accordingly too
    - Note: this means the image and target tensors will then be of size `[n_samples * IMG_DUP,]`

## Targets

Target data, for now, includes energy and total width (gamma) in a tensor of size `[n_samples, 2]`.
- Energy of n-th sample: `[n, 0]`
- Gamma of n-th sample: `[n, 1]`

# Model

There are two models:
- ResonanceCNN (`model.py`) -- A normal CNN run on 1-channel image data
- ResonancePartialCNN (`partial_model.py`) -- A partial CNN run on 2-channel image data, where channel 2 is a visibility mask
    - Convolutions are weighted by the proportion of visible parameters in a sliding window
    - Windows with a small amount of visible parameters are weighted higher, i.e. if `K` is the number of parameters in a window, and `valid_count` is the number of those parameters that match to a `1` in the mask channel, then the convolution is scaled by `K / valid_count`
    - Non-zero parameters that match to a `0` in the mask channel are counted as `0`

# Output

For now, the model only predicts the energy level of detected resonances. Soon (hopefully) it will be extended to predict total width, spin, parity, partial widths ...

## Training data
Training data, including train/val loss and train/val MAE over all epochs, are saved as .csv files in `data/training_data`.