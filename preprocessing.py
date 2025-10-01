import json
import gzip
import numpy as np
import pandas as pd
import torch

global_E_min = 8.0
global_E_max = 20.0
global_E_step = 0.1

global_A_min = 25.0
global_A_max = 175.0
global_A_step = 10.0

IMG_DUP = 10

def global_grid():
    
    E_axis = np.linspace(global_E_min, \
                         global_E_max, \
                         int((global_E_max - global_E_min) / global_E_step) + 1)
    A_axis = np.linspace(global_A_min, \
                         global_A_max, \
                         int((global_A_max - global_A_min) / global_A_step) + 1)

    return E_axis, A_axis

def place_image_on_grid(E_vals, A_vals, cx_vals):

    E_vals = np.array(E_vals)
    A_vals = np.array(A_vals)
    cx_vals = np.array(cx_vals)

    E_axis, A_axis = global_grid()
    num_E = len(E_axis)
    num_A = len(A_axis)

    image = torch.zeros((2, num_E, num_A))

    # get positions of nearest coordinates on grid
    E_idx = np.abs(E_vals[:, None] - E_axis[None, :]).argmin(axis=1)
    A_idx = np.abs(A_vals[:, None] - A_axis[None, :]).argmin(axis=1)

    for i in range(len(cx_vals)):
        ei, ai = E_idx[i], A_idx[i]
        image[0, ei, ai] = cx_vals[i]
        image[1, ei, ai] = 1.0 # mask set to one initially

    return image

def random_grid(strength):

    E_min, E_max = random_range(global_E_min, \
                                        global_E_max, \
                                        global_E_step, \
                                        strength)
    
    A_min, A_max = random_range(global_A_min, \
                                        global_A_max, \
                                        global_A_step, \
                                        strength)

    E_axis = np.linspace(E_min, \
                         E_max, \
                         int((E_max - E_min) / global_E_step) + 1)
    A_axis = np.linspace(A_min, \
                         A_max, \
                         int((A_max - A_min) / global_A_step) + 1)

    return E_axis, A_axis

def crop_image(image, strength):

    if strength == 0.0:
        image[1, :, :] = 1.0
        return image

    image[1, :, :] = 0.0 # set whole mask to zero

    global_E_axis, global_A_axis = global_grid()
    random_E_axis, random_A_axis = random_grid(strength)

    E_idx = np.searchsorted(global_E_axis, random_E_axis)
    A_idx = np.searchsorted(global_A_axis, random_A_axis)

    for E in E_idx:
        for A in A_idx:
            image[1, E, A] = 1.0

    return image

def random_range(minimum, maximum, step, strength):

    axis = np.linspace(minimum, \
                         maximum, \
                         int((maximum - minimum) / step) + 1)
    
    n = len(axis)

    subset_length_minimum = n * (0.5 * (1 - strength))
    subset_length_maximum = n * (1 - strength)

    subset_length = int(np.random.uniform(subset_length_minimum, subset_length_maximum))

    minimum_upper_limit = maximum - (subset_length * step)
    
    minimum_ = round(np.random.uniform(minimum, minimum_upper_limit), 1)
    maximum_ = round(minimum_ + (subset_length * step), 1)

    return minimum_, maximum_

def sobel(image):
    
    device = image.device
    dtype = image.dtype

    x = image[0].unsqueeze(0).unsqueeze(0)

    kx = torch.tensor([[-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]], dtype=dtype, device=device).view(1, 1, 3, 3)
    ky = torch.tensor([[-1, -2, -1],
                       [ 0,  0,  0],
                       [ 1,  2,  1]], dtype=dtype, device=device).view(1, 1, 3, 3)

    gx = F.conv2d(x, kx, padding=1)
    gy = F.conv2d(x, ky, padding=1)

    grad = torch.sqrt(gx * gx + gy * gy + 1e-12)
    out = image.clone()
    out[0] = grad[0]

    return out