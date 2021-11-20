import numpy as np
import cv2
import torch.nn.functional as F
import random
from scipy import special
import torch


def generate_gaussian_noise(img, sigma=10, gray_noise=False):
    """Generate Gaussian noise.
    Args:
        img (Numpy array): Input image, shape (h, w, c), range [0, 1], float32.
        sigma (float): Noise scale (measured in range 255). Default: 10.
    Returns:
        (Numpy array): Returned noisy image, shape (h, w, c), range[0, 1],
            float32.
    """
    if gray_noise:
        noise = np.float32(np.random.randn(*(img.shape[0:2]))) * sigma / 255.
        noise = np.expand_dims(noise, axis=2).repeat(3, axis=2)
    else:
        noise = np.float32(np.random.randn(*(img.shape))) * sigma / 255.
    return noise
    
def random_generate_gaussian_noise(img, sigma_range=(0, 10), gray_prob=0):
    sigma = np.random.uniform(sigma_range[0], sigma_range[1])
    if np.random.uniform() < gray_prob:
        gray_noise = True
    else:
        gray_noise = False
    return generate_gaussian_noise(img, sigma, gray_noise)

def random_add_gaussian_noise(img, sigma_range=(0, 1.0), gray_prob=0, clip = False, rounds = False):
    noise = random_generate_gaussian_noise(img, sigma_range, gray_prob)
    out = img + torch.tensor(noise)
    if clip and rounds:
        out = np.clip((out * 255.0).round(), 0, 255) / 255.
    elif clip:
        out = np.clip(out, 0, 1)
    elif rounds:
        out = (out * 255.0).round() / 255.
    return out


def generate_poisson_noise(img, scale=1.0, gray_noise=False):
    if gray_noise:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # round and clip image for counting vals correctly
    img = np.clip((img * 255.0).round(), 0, 255) / 255.
    vals = len(np.unique(img))
    vals = 2**np.ceil(np.log2(vals))
    out = np.float32(np.random.poisson(img * vals) / float(vals))
    noise = torch.tensor(out) - img
    if gray_noise:
        noise = np.repeat(noise[:, :, np.newaxis], 3, axis=2)
    return noise * scale

def random_generate_poisson_noise(img, scale_range=(0, 1.0), gray_prob=0):
    scale = np.random.uniform(scale_range[0], scale_range[1])
    if np.random.uniform() < gray_prob:
        gray_noise = True
    else:
        gray_noise = False
    return generate_poisson_noise(img, scale, gray_noise)


def random_add_poisson_noise(img, scale_range=(0, 1.0), gray_prob=0, clip = False, rounds=False):
    noise = random_generate_poisson_noise(img, scale_range, gray_prob)
    out = img + torch.tensor(noise)
    if clip and rounds:
        out = np.clip((out * 255.0).round(), 0, 255) / 255.
    elif clip:
        out = np.clip(out, 0, 1)
    elif rounds:
        out = (out * 255.0).round() / 255.
    return out

def filter2D(img, kernel):
    """PyTorch version of cv2.filter2D
    Args:
        img (Tensor): (b, c, h, w)
        kernel (Tensor): (b, k, k)
    """
    k = kernel.size(-1)
    b, c, h, w = img.size()
    if k % 2 == 1:
        img = F.pad(img, (k // 2, k // 2, k // 2, k // 2), mode='reflect')
    else:
        raise ValueError('Wrong kernel size')

    ph, pw = img.size()[-2:]

    if kernel.size(0) == 1:
        # apply the same kernel to all batch images
        img = img.view(b * c, 1, ph, pw)
        kernel = kernel.view(1, 1, k, k)
        return F.conv2d(img, kernel, padding=0).view(b, c, h, w)
    else:
        img = img.view(1, b * c, ph, pw)
        kernel = kernel.view(b, 1, k, k).repeat(1, c, 1, 1).view(b * c, 1, k, k)
        return F.conv2d(img, kernel, groups=b * c).view(b, c, h, w)


def circular_lowpass_kernel(cutoff, kernel_size, pad_to=0):
    """2D sinc filter, ref: https://dsp.stackexchange.com/questions/58301/2-d-circularly-symmetric-low-pass-filter
    Args:
        cutoff (float): cutoff frequency in radians (pi is max)
        kernel_size (int): horizontal and vertical size, must be odd.
        pad_to (int): pad kernel size to desired size, must be odd or zero.
    """
    assert kernel_size % 2 == 1, 'Kernel size must be an odd number.'
    kernel = np.fromfunction(
        lambda x, y: cutoff * special.j1(cutoff * np.sqrt(
            (x - (kernel_size - 1) / 2)**2 + (y - (kernel_size - 1) / 2)**2)) / (2 * np.pi * np.sqrt(
                (x - (kernel_size - 1) / 2)**2 + (y - (kernel_size - 1) / 2)**2)), [kernel_size, kernel_size])
    kernel[(kernel_size - 1) // 2, (kernel_size - 1) // 2] = cutoff**2 / (4 * np.pi)
    kernel = kernel / np.sum(kernel)
    if pad_to > kernel_size:
        pad_size = (pad_to - kernel_size) // 2
        kernel = np.pad(kernel, ((pad_size, pad_size), (pad_size, pad_size)))
    return kernel

def filter2D(img, kernel):
    """PyTorch version of cv2.filter2D
    Args:
        img (Tensor): (b, c, h, w)
        kernel (Tensor): (b, k, k)
    """
    k = kernel.size(-1)
    b, c, h, w = img.size()
    if k % 2 == 1:
        img = F.pad(img, (k // 2, k // 2, k // 2, k // 2), mode='reflect')
    else:
        raise ValueError('Wrong kernel size')

    ph, pw = img.size()[-2:]

    if kernel.size(0) == 1:
        # apply the same kernel to all batch images
        img = img.view(b * c, 1, ph, pw)
        kernel = kernel.view(1, 1, k, k)
        return F.conv2d(img, kernel, padding=0).view(b, c, h, w)
    else:
        img = img.view(1, b * c, ph, pw)
        kernel = kernel.view(b, 1, k, k).repeat(1, c, 1, 1).view(b * c, 1, k, k)
        return F.conv2d(img, kernel, groups=b * c).view(b, c, h, w)

def sinc_filter(img):
    kernel_range = [2 * v + 1 for v in range(3, 11)] 
    kernel_size = random.choice(kernel_range)
    omega_c = np.random.uniform(np.pi / 3, np.pi)
    sinc_kernel = torch.tensor(circular_lowpass_kernel(omega_c, kernel_size, pad_to=21))
    simg = filter2D(torch.tensor(img).unsqueeze(0), sinc_kernel.float())

    return simg
