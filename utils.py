import scipy.ndimage as ndimage
import numpy as np
import torch
from surface_distance.metrics import compute_robust_hausdorff, compute_surface_distances
from tqdm import trange

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    
def compute_sdm(mask):
    #check if mask is empty
    if mask.sum() == 0:
        return torch.zeros_like(mask, dtype=torch.float32)
    mask_np = mask.cpu().numpy()
    dist_outside = ndimage.distance_transform_edt(mask_np == 0)  # Distance to nearest 1 (inside object)
    dist_inside = ndimage.distance_transform_edt(mask_np == 1)   # Distance to nearest 0 (outside object)
    sdm = dist_outside - dist_inside
    
    return torch.tensor(sdm, dtype=torch.float32)


def smooth_heaviside_tanh(sdm, epsilon=20):
    return (1 + torch.tanh(sdm / epsilon)) - 1


def multilabel_sdm(seg):
    edm = torch.zeros_like(seg, dtype=torch.float32)
    for i in range(seg.shape[0]):
        edm[i] = compute_sdm(seg[i])
        #addidtional heavside step
        edm[i] = smooth_heaviside_tanh(edm[i])
    return edm


def dice_coeff(outputs, labels, max_label):
    dice = torch.FloatTensor(max_label-1).fill_(0)
    for label_num in range(1, max_label):
        iflat = (outputs==label_num).view(-1).float()
        tflat = (labels==label_num).view(-1).float()
        intersection = torch.mean(iflat * tflat)
        dice[label_num-1] = (2. * intersection) / (1e-8 + torch.mean(iflat) + torch.mean(tflat))
    return dice


def evaluate(model, mesh, latent_codes, ssms, H, W, C, mode='less'):
    N = ssms.shape[0] 
    HDs = torch.zeros(N,C)
    dices = torch.zeros(N,C)
    for i in trange(N):
        out = model(torch.cat([mesh[0].view(-1,2), latent_codes[i].unsqueeze(0).repeat(H*W,1)], dim=-1)).cpu().detach()
        for j in range(C):
            if mode == 'less':
                dices[i,j] = dice_coeff(out.view(H,W,-1)[...,j]<0, ssms[i].view(H,W,-1)[...,j]<0, 2)
                q = compute_robust_hausdorff(compute_surface_distances((out.view(H,W,-1)[...,j]<0).cpu().numpy(), (ssms[i].view(H,W,-1)[...,j]<0).cpu().numpy(), (0.7,0.7)),95)
            elif mode == 'greater':
                dices[i,j] = dice_coeff(out.view(H,W,-1)[...,j]>0, ssms[i].view(H,W,-1)[...,j]>0, 2)
                q = compute_robust_hausdorff(compute_surface_distances((out.view(H,W,-1)[...,j]>0).cpu().numpy(), (ssms[i].view(H,W,-1)[...,j]>0).cpu().numpy(), (0.7,0.7)),95)
            HDs[i,j] = q
    return HDs, dices

def visualize_differences(A, B):
    if isinstance(A, torch.Tensor):
        A = A.cpu().numpy()
    if isinstance(B, torch.Tensor):
        B = B.cpu().numpy()

    H, W, num_labels = A.shape
    diff_image = np.zeros((H, W, 3), dtype=np.uint8)

    colors = [
        [255, 0, 0],   
        [0, 255, 0],   
        [0, 0, 255],    
        [255, 255, 0],  
        [255, 0, 255]   
    ]

    for i in range(num_labels):
        diff_mask = (A[:, :, i] != B[:, :, i])
        for j in range(3):  
            diff_image[:, :, j] += (diff_mask * colors[i][j]).astype(np.uint8)

    diff_image = np.clip(diff_image, 0, 255)

    return diff_image

def compute_metrics_single(reference, prediction, mode='less'):
    #reference HxWxC, prediction HxWxC
    H, W, C = reference.shape 
    dices = torch.zeros(C)
    HDs = torch.zeros(C)
    for i in range(C):
        if mode == 'less':
            dices[i] = dice_coeff(reference.view(H,W,-1)[...,i]<0, prediction.view(H,W,-1)[...,i]<0,2)
            HDs[i] = compute_robust_hausdorff(compute_surface_distances((reference.view(H,W,-1)[...,i]<0).cpu().numpy(), (prediction.view(H,W,-1)[...,i]<0).cpu().numpy(), (0.7,0.7)),95)
        elif mode == 'greater':
            dices[i] = dice_coeff(reference.view(H,W,-1)[...,i]>0, prediction.view(H,W,-1)[...,i]>0,2)
            HDs[i] = compute_robust_hausdorff(compute_surface_distances((reference.view(H,W,-1)[...,i]>0).cpu().numpy(), (prediction.view(H,W,-1)[...,i]>0).cpu().numpy(), (0.7,0.7)),95)
    return dices, HDs