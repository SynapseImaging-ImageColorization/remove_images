import cv2
import numpy as np
import torch
from scipy.linalg import sqrtm
from torch.backends import mps
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.image.fid import FrechetInceptionDistance

device = torch.device(
    'cuda' if torch.cuda.is_available() else
    'mps' if mps.is_available()
    else 'cpu'
)


def calculate_psnr(img1, img2):
    img1 = torch.from_numpy(img1).permute(2, 0, 1).unsqueeze(0).float().to(device)
    img2 = torch.from_numpy(img2).permute(2, 0, 1).unsqueeze(0).float().to(device)

    mse = torch.mean((img1 - img2) ** 2)
    psnr = 20 * torch.log10(255.0 / torch.sqrt(mse))
    return psnr.item()


def calculate_ssim(img1, img2):
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    img1 = torch.from_numpy(img1).float().to(device)
    img2 = torch.from_numpy(img2).float().to(device)

    img1 = img1.unsqueeze(0).unsqueeze(0)
    img2 = img2.unsqueeze(0).unsqueeze(0)

    score = structural_similarity_index_measure(img1, img2)
    return score.item()


def calculate_fid(img1, img2):
    img1 = img1.reshape((img1.shape[0], -1))
    img2 = img2.reshape((img2.shape[0], -1))
    mu1, sigma1 = img1.mean(axis=0), np.cov(img1, rowvar=False)
    mu2, sigma2 = img2.mean(axis=0), np.cov(img2, rowvar=False)

    ssdiff = np.sum((mu1 - mu2)**2.0)
    covmean = sqrtm(sigma1.dot(sigma2))

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid
