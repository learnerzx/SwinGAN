import yaml
from types import SimpleNamespace
import logging
import os
import utils.metrics as mt
import numpy as np
import torch
from math import log10
import glob
import h5py
from pprint import pprint
import pickle
import cv2
import scipy
from Networks.generator_model import WNet
from utils.data_vis import plot_imgs,save_imgs
from utils.data_save import save_data

from scipy.io import loadmat, savemat

def crop_toshape(kspace_cplx, args):
    if kspace_cplx.shape[0] == args.img_size:
        return kspace_cplx
    if kspace_cplx.shape[0] % 2 == 1:
        kspace_cplx = kspace_cplx[:-1, :-1]
    crop = int((kspace_cplx.shape[0] - args.img_size) / 2)
    kspace_cplx = kspace_cplx[crop:-crop, crop:-crop]

    return kspace_cplx


def fft2(img):
    return np.fft.fftshift(np.fft.fft2(img))

def ifft2(kspace_cplx):
    return np.absolute(np.fft.ifft2(kspace_cplx))[None, :, :]


def slice_preprocess(kspace_cplx, slice_num, masks, maskedNot, args):
    # crop to fix size
    kspace_cplx = crop_toshape(kspace_cplx, args)
    # split to real and imaginary channels
    kspace = np.zeros((args.img_size, args.img_size, 2))
    kspace[:, :, 0] = np.real(kspace_cplx).astype(np.float32)
    kspace[:, :, 1] = np.imag(kspace_cplx).astype(np.float32)
    # target image:
    image = ifft2(kspace_cplx)

    # HWC to CHW
    kspace = kspace.transpose((2, 0, 1))
    masked_Kspace = kspace * masks
    masked_Kspace += np.random.uniform(low=args.minmax_noise_val[0], high=args.minmax_noise_val[1],
                                       size=masked_Kspace.shape) * maskedNot

    return masked_Kspace, kspace, image


def get_args():
    with open('config.yaml') as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    args = SimpleNamespace(**data)

    args.masked_kspace = True
    if args.mask_type=='random':
        args.mask_path = './Masks/random/mask_{}_{}.pickle'.format(args.sampling_percentage,args.img_size)
    else:
        args.mask_path = './Masks/{}/{}_{}_{}_{}.mat'.format(args.mask_type,args.mask_type, args.img_size,args.img_size,args.sampling_percentage)

    pprint(data)


    return args
def mean_squared_error(image0, image1):
    """
    Compute the mean-squared error between two images.

    Parameters
    ----------
    image0, image1 : ndarray
        Images.  Any dimensionality, must have same shape.

    Returns
    -------
    mse : float
        The mean-squared error (MSE) metric.

    Notes
    -----
    .. versionchanged:: 0.16
        This function was renamed from ``skimage.measure.compare_mse`` to
        ``skimage.metrics.mean_squared_error``.

    """
    check_shape_equality(image0, image1)
    image0, image1 = _as_floats(image0, image1)
    return np.mean((image0 - image1) ** 2, dtype=np.float64)

def check_shape_equality(im1, im2):
    """Raise an error if the shape do not match."""
    if not im1.shape == im2.shape:
        raise ValueError('Input images must have the same dimensions.')
    return
def _as_floats(image0, image1):
    """
    Promote im1, im2 to nearest appropriate floating point precision.
    """
    float_type = np.result_type(image0.dtype, image1.dtype, np.float32)
    image0 = np.asarray(image0, dtype=float_type)
    image1 = np.asarray(image1, dtype=float_type)
    return image0, image1



if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    # Set device and GPU (currently only single GPU training is supported
    logging.info(f'Using device {args.device}')
    if args.device == 'cuda':
        logging.info(f'Using GPU {args.gpu_id}')
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    nmse_meter = mt.AverageMeter()
    psnr_meter = mt.AverageMeter()
    ssim_meter = mt.AverageMeter()
    zf_psnr_meter=mt.AverageMeter()
    zf_ssim_meter=mt.AverageMeter()
    # Load network
    logging.info("Loading model {}".format(args.model))
    net = WNet(args, masked_kspace=args.masked_kspace)
    net.to(device=args.device)

    checkpoint = torch.load(args.model, map_location=args.device)
    net.load_state_dict(checkpoint['G_model_state_dict'])
    net.eval()

    logging.info("Model loaded !")


    # with open(args.mask_path, 'rb') as pickle_file:
        # masks_dictionary = pickle.load(pickle_file)
    # masks_dictionary=loadmat(args.mask_path)
    if args.mask_type == 'radial' and args.sampling_percentage == 30:
        with open(args.mask_path, 'rb') as pickle_file:
            masks_dictionary = pickle.load(pickle_file)
            # self.mask = torch.tensor(masks['mask1'] == 1, device=self.args.device)
            masks = masks_dictionary
            maskNot = 1 - masks_dictionary

    elif args.mask_type == 'radial' and args.sampling_percentage == 50:
        mask_shift = cv2.imread(r'E:\code\code_backup\Masks\radial\radial_50.tif', 0) / 255
        # mask = scipy.ifft(mask_shift)
        masks = mask_shift
        maskNot = 1 - masks
    elif args.mask_type == 'random':
        with open(args.mask_path, 'rb') as pickle_file:
            masks_dictionary = pickle.load(pickle_file)
            masks = masks_dictionary['mask1']
            maskNot = 1 - masks_dictionary['mask1']
            # masks = masks_dictionary
            # maskNot = 1 - masks_dictionary
    else:
        masks_dictionary = loadmat(args.mask_path)
        try:
            masks = masks_dictionary['Umask']
            maskNot = 1 - masks_dictionary['Umask']
        except:
            try:
                masks = masks_dictionary['maskRS2']
                maskNot = 1 - masks_dictionary['maskRS2']
            except:
                masks = masks_dictionary['population_matrix']
                maskNot = 1 - masks_dictionary['population_matrix']

    # masks = np.dstack((masks_dictionary['mask0'], masks_dictionary['mask1'], masks_dictionary['mask2']))


    # masks = masks_dictionary['Umask']
    #
    # maskNot = 1 - masks_dictionary['Umask']

    total_nmse=[]
    total_psnr=[]
    total_ssim=[]
    total_zf_psnr=[]
    total_zf_ssim=[]
    slice_begin=args.slice_range[0]
    slice_end=args.slice_range[-1]
    test_files = glob.glob(os.path.join(args.predict_data_dir, '*.hdf5'))
    for i, infile in enumerate(test_files):
        # logging.info("\nPredicting image {} ...".format(infile))

        with h5py.File(infile, 'r') as f:
            # img_shape = f['data'].shape
            img_shape =(256,256,80)
            fully_sampled_imgs = f['data'][:,:,slice_begin:slice_end]
            # fully_sampled_imgs=fully_sampled_imgs.transpose(2,0,1)
        #Preprocess data:
        rec_imgs = np.zeros(img_shape)
        rec_Kspaces = np.zeros(img_shape, dtype=np.csingle) #complex
        F_rec_Kspaces = np.zeros(img_shape)
        ZF_img = np.zeros(img_shape)
        refine_Imgs=np.zeros(img_shape)
        for slice_num in range(slice_begin,img_shape[2]+slice_begin):
            add = int(args.num_input_slices / 2)
            with h5py.File(infile, 'r') as f:
                if slice_num == slice_begin:
                    imgs = np.dstack((f['data'][:, :, slice_begin], f['data'][:, :, slice_begin], f['data'][:, :, slice_begin+1]))
                elif slice_num == img_shape[2]+slice_begin-1:
                    imgs = np.dstack((f['data'][:, :, slice_num-1], f['data'][:, :, slice_num], f['data'][:, :, slice_num]))
                else:
                    imgs = np.dstack((f['data'][:, :, slice_num-1], f['data'][:, :, slice_num], f['data'][:, :, slice_num + 1]))

            masked_Kspaces_np = np.zeros((args.num_input_slices * 2, args.img_size, args.img_size))
            target_Kspace = np.zeros((2, args.img_size, args.img_size))
            target_img = np.zeros((1, args.img_size, args.img_size))

            for slice_j in range(args.num_input_slices):
                img = imgs[:, :, slice_j]
                kspace = fft2(img)
                slice_masked_Kspace, slice_full_Kspace, slice_full_img = slice_preprocess(kspace, slice_j,
                                                                                          masks, maskNot, args)
                masked_Kspaces_np[slice_j * 2:slice_j * 2 + 2, :, :] = slice_masked_Kspace
                if slice_j == int(args.num_input_slices / 2):
                    target_Kspace = slice_full_Kspace
                    target_img = slice_full_img

            masked_Kspaces = np.expand_dims(masked_Kspaces_np, axis=0)

            masked_Kspaces = torch.from_numpy(masked_Kspaces).to(device=args.device, dtype=torch.float32)

            #Predict:
            rec_img, rec_Kspace, F_rec_Kspace = net(masked_Kspaces)

            rec_img = np.squeeze(rec_img.data.cpu().numpy())
            rec_Kspace = np.squeeze(rec_Kspace.data.cpu().numpy())


            rec_Kspace = (rec_Kspace[0, :, :] + 1j*rec_Kspace[1, :, :])

            F_rec_Kspace = np.squeeze(F_rec_Kspace.data.cpu().numpy())

            rec_imgs[:, :, slice_num-slice_begin] = rec_img
            rec_Kspaces[:, :, slice_num-slice_begin] = rec_Kspace
            F_rec_Kspaces[:, :, slice_num-slice_begin] = F_rec_Kspace
            ZF_img[:, :, slice_num-slice_begin] = np.squeeze(ifft2((masked_Kspaces_np[2, :, :] + 1j*masked_Kspaces_np[3, :, :])))

            fully_sampled_img =fully_sampled_imgs[:, :, slice_num-slice_begin]
            nmse=mt.nmse(fully_sampled_img,rec_img)
            psnr=mt.psnr(fully_sampled_img,rec_img)
            ssim=mt.ssim(fully_sampled_img,rec_img)

            zf_psnr=mt.psnr(fully_sampled_img,ZF_img[:,:,slice_num-slice_begin])
            zf_ssim=mt.ssim(fully_sampled_img,ZF_img[:,:,slice_num-slice_begin])
            # mse=mean_squared_error(fully_sampled_img,rec_img)
            # psnr=10 * log10(fully_sampled_img.max()**2/ mse)
            if nmse and psnr and ssim and zf_psnr and zf_ssim!=np.inf:

                nmse_meter.update(nmse, 1)
                psnr_meter.update(psnr, 1)
                ssim_meter.update(ssim, 1)
                zf_psnr_meter.update(zf_psnr, 1)        #########
                zf_ssim_meter.update(zf_ssim, 1)        #########

        total_nmse.append(nmse_meter.avg)
        total_psnr.append(psnr_meter.avg)
        total_ssim.append(ssim_meter.avg)
        total_zf_psnr.append(zf_psnr_meter.avg)        #########
        total_zf_ssim.append(zf_ssim_meter.avg)        #########

        print("==> Evaluate Metric of image{}".format(infile))
        print("Results ----------")
        print("NMSE: {:.4}".format(nmse_meter.avg))
        print("PSNR: {:.4}".format(psnr_meter.avg))
        print("SSIM: {:.4}".format(ssim_meter.avg))
        print("ZF_PSNR: {:.4}".format(zf_psnr_meter.avg))
        print("ZF_SSIM: {:.4}".format(zf_ssim_meter.avg))
        print("------------------")

        if args.save_prediction:
            os.makedirs(args.save_path, exist_ok=True)
            out_file_name = args.save_path +'/'+os.path.split(infile)[1]
            save_data(out_file_name, rec_imgs, F_rec_Kspaces, fully_sampled_imgs, ZF_img, rec_Kspaces)
            save_imgs(rec_imgs, F_rec_Kspaces, fully_sampled_imgs, ZF_img)
            logging.info("reconstructions save to: {}".format(out_file_name))

        if args.visualize_images:
            logging.info("Visualizing results for image {}, close to continue ...".format(infile))
            plot_imgs(rec_imgs, F_rec_Kspaces,fully_sampled_imgs, ZF_img)



    print("avg_NMSE: {:.4}".format(np.mean(total_nmse)))
    print("avg_PSNR: {:.4}".format(np.mean(total_psnr)))
    print("avg_SSIM: {:.4}".format(np.mean(total_ssim)))

    print("std_NMSE: {:.4}".format(np.std(total_nmse)))
    print("std_PSNR: {:.4}".format(np.std(total_psnr)))
    print("std_SSIM: {:.4}".format(np.std(total_ssim)))

    print("avg_ZF_PSNR: {:.4}".format(np.mean(total_zf_psnr)))
    print("avg_ZF_SSIM: {:.4}".format(np.mean(total_zf_ssim)))
    print("std_ZF_PSNR: {:.4}".format(np.std(total_zf_psnr)))
    print("std_ZF_SSIM: {:.4}".format(np.std(total_zf_ssim)))
