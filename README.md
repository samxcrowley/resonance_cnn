There are two models:
- ResonanceCNN (`model.py`) -- This is just a normal CNN run on 1-channel image data
- Resonance_PartialCNN (`partial_model.py`) -- This is a partial CNN run on 2-channel image data, where channel 2 is a visibility mask

Image data are saved to `data/images`. Images are cropped according to parameters `crop_coef` and `angle_p`. Energy ranges at each angle are cropped from the bottom and top, each by up to 1/`crop_coef`. `angle_p` is the percentage of angles dropped from the images at random.

Training data are saved as .csv files in `data/training_data`.