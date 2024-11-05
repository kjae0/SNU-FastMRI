
import os
import h5py
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def seed_everything(seed):
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False  # True can improve train speed
        torch.backends.cudnn.deterministic = True  # Guarantee that the convolution algorithm returned each time will be deterministic
        # torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def save_reconstructions(reconstructions, out_dir, targets=None, inputs=None, kspace=None):
    """
    Saves the reconstructions from a model into h5 files that is appropriate for submission
    to the leaderboard.

    Args:
        reconstructions (dict[str, np.array]): A dictionary mapping input filenames to
            corresponding reconstructions (of shape num_slices x height x width).
        out_dir (pathlib.Path): Path to the output directory where the reconstructions
            should be saved.
        target (np.array): target array
    """
    out_dir.mkdir(exist_ok=True, parents=True)
    for fname, recons in reconstructions.items():
        with h5py.File(out_dir / fname, 'w') as f:
            f.create_dataset('reconstruction', data=recons)
            if kspace is not None:
                f.create_dataset('kspace', data=kspace[fname])
            if targets is not None:
                f.create_dataset('target', data=targets[fname])
            if inputs is not None:
                f.create_dataset('input', data=inputs[fname])


# Utility and helper functions for MRAugment.

def _to_repeated_list(a, length):
    if isinstance(a, list):
        return a
    elif isinstance(a, tuple):
        return list(a)
    else:
        a = [a] * length
        return a
    
def pad_if_needed(im, min_shape, mode):
    min_shape = _to_repeated_list(min_shape, 2)
    if im.shape[-2] >= min_shape[0] and im.shape[-1] >= min_shape[1]:
        return im
    else:
        pad = [0, 0]
        if im.shape[-2] < min_shape[0]:
            p = (min_shape[0] - im.shape[-2])//2 + 1
            pad[0] = p
        if im.shape[-1] < min_shape[1]:
            p = (min_shape[1] - im.shape[-1])//2 + 1
            pad[1] = p
        if len(im.shape) == 2:
            pad = ((pad[0], pad[0]), (pad[1], pad[1]))
        else:
            assert len(im.shape) == 3
            pad = ((0, 0), (pad[0], pad[0]), (pad[1], pad[1]))

        padded = np.pad(im, pad_width=pad, mode=mode)
        return padded
    
def crop_if_needed(im, max_shape):
    assert len(max_shape) == 2
    if im.shape[-2] >= max_shape[0]:
        h_diff = im.shape[-2] - max_shape[0]
        h_crop_before = h_diff // 2
        h_interval = max_shape[0]
    else:
        h_crop_before = 0
        h_interval = im.shape[-2]

    if im.shape[-1] >= max_shape[1]:
        w_diff = im.shape[-1] - max_shape[1]
        w_crop_before = w_diff // 2
        w_interval = max_shape[1]
    else:
        w_crop_before = 0
        w_interval = im.shape[-1]

    return im[...,h_crop_before:h_crop_before+h_interval, w_crop_before:w_crop_before+w_interval]

def complex_crop_if_needed(im, max_shape):
    assert len(max_shape) == 2
    if im.shape[-3] >= max_shape[0]:
        h_diff = im.shape[-3] - max_shape[0]
        h_crop_before = h_diff // 2
        h_interval = max_shape[0]
    else:
        h_crop_before = 0
        h_interval = im.shape[-3]

    if im.shape[-2] >= max_shape[1]:
        w_diff = im.shape[-2] - max_shape[1]
        w_crop_before = w_diff // 2
        w_interval = max_shape[1]
    else:
        w_crop_before = 0
        w_interval = im.shape[-2]

    return im[...,h_crop_before:h_crop_before+h_interval, w_crop_before:w_crop_before+w_interval, :]
    
    
def ifft2_np(x):
    return np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(x.astype(np.complex64), axes=[-2, -1]), norm='ortho'), axes=[-2, -1]).astype(np.complex64)


def fft2_np(x):
    return np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(x.astype(np.complex64), axes=[-2, -1]), norm='ortho'), axes=[-2, -1]).astype(np.complex64)


def complex_channel_first(x):
    assert x.shape[-1] == 2
    if len(x.shape) == 3:
        # Single-coil (H, W, 2) -> (2, H, W)
        x = x.permute(2, 0, 1)
    else:
        # Multi-coil (C, H, W, 2) -> (2, C, H, W)
        assert len(x.shape) == 4
        x = x.permute(3, 0, 1, 2)
    return x

def complex_channel_last(x):
    assert x.shape[0] == 2
    if len(x.shape) == 3:
        # Single-coil (2, H, W) -> (H, W, 2)
        x = x.permute(1, 2, 0)
    else:
        # Multi-coil (2, C, H, W) -> (C, H, W, 2)
        assert len(x.shape) == 4
        x = x.permute(1, 2, 3, 0)
    return x
