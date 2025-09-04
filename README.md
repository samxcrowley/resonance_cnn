There are two models:
- ResonanceCNN (`model.py`) -- This is just a normal CNN run on 1-channel image data
- ResonancePartialCNN (`partial_model.py`) -- This is a partial CNN run on 2-channel image data, where channel 2 is a visibility mask
    - Convolutions are weighted by the proportion of visible parameters in a sliding window
    - Windows with a small amount of visible parameters are weighted higher, i.e. if `K` is the number of parameters in a window, and `valid_count` is the number of those parameters that match to a `1` in the mask channel, then the convolution is scaled by `K / valid_count`
    - Non-zero parameters that match to a `0` in the mask channel are counted as a `0`

Image data are saved to `data/images`. Images are cropped according to parameters `crop_coef` and `angle_p`. Energy ranges at each angle are cropped from the bottom and top, each by up to `1 / crop_coef`. `angle_p` is the percentage of angles dropped from the images at random.

Training data are saved as .csv files in `data/training_data`.