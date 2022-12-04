import torch
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def nmse(gt, pred):
    """Compute Normalized Mean Squared Error (NMSE)"""
    return np.linalg.norm(gt - pred) ** 2 / np.linalg.norm(gt) ** 2


def psnr(gt, pred):
    """Compute Peak Signal to Noise Ratio metric (PSNR)"""
    return peak_signal_noise_ratio(gt, pred, data_range=gt.max())


def psnr_slice(gt, pred, maxval=None):
    assert type(gt) == type(pred)
    if type(pred) is torch.Tensor:
        gt, pred = gt.detach().cpu().numpy(), pred.detach().cpu().numpy()
    batch_size = gt.shape[0]
    PSNR = 0.0
    for i in range(batch_size):
        max_val = gt[i].max() if maxval is None else maxval
        PSNR += peak_signal_noise_ratio(gt[i].squeeze(), pred[i].squeeze(), data_range=max_val)
    return PSNR / batch_size

def ssim(gt, pred, maxval=None):
    assert type(gt) == type(pred)
    if type(pred) is torch.Tensor:
        gt, pred = gt.detach().cpu().numpy(), pred.detach().cpu().numpy()
        batch_size = gt.shape[0]
        SSIM = 0.0
        for i in range(batch_size):
            max_val = gt[i].max() if maxval is None else maxval
            if maxval==0:
                max_val=1.0
            SSIM += structural_similarity(gt[i].squeeze(), pred[i].squeeze(), data_range=max_val)
        SSIM = SSIM / batch_size
    else:
        max_val = gt.max() if maxval is None else maxval
        SSIM = structural_similarity(gt, pred, data_range=1.0)

    return SSIM

# def ssim(gt, pred, maxval=None):
#     """Compute Structural Similarity Index Metric (SSIM)"""
#     if(gt.dtype.__str__() not in {"float64","float32"}):
#         # maxval = gt.cpu().numpy().max() if maxval is None else maxval
#
#         ssim = 0
#         for slice_num in range(gt.shape[0]):
#             gt1=gt[slice_num].cpu().numpy().transpose(1,2,0).squeeze()
#             pred1=pred[slice_num].cpu().numpy().transpose(1,2,0).squeeze()
#             ssim = ssim + structural_similarity(
#                 gt1, pred1,data_range=gt1.max()
#             )
#
#         ssim=ssim/gt.shape[0]
#     else:
#         maxval =gt.max()
#         ssim = structural_similarity(
#             gt, pred,data_range=maxval
#         )
#     # skimage.measure.compare_ssim(gt, pred)
#     return ssim

class AverageMeter(object):
    """Computes and stores the average and current value.

       Code imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count