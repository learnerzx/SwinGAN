from matplotlib import pyplot as plt
import matplotlib
import numpy as np
from IPython.display import clear_output
from PIL import Image

# def plot_imgs(rec_imgs, F_rec_Kspaces,refine_Img, fully_sampled_img, ZF_img):
#     # plt.figure()
#
#     slices = [0, 40, 50, 60, 70, 79]
#
#     # for slice in range(rec_imgs.shape[2])
#     for slice in slices:
#         fig, ax = plt.subplots(1, 5, figsize=(40, 10))
#         plt.subplots_adjust(hspace=0, wspace=0)
#         ax[0].set_title('Final ', fontsize=30)
#         ax[0].imshow(rec_imgs[:, :, slice], vmin=0, vmax=1, cmap=plt.get_cmap('gray'))
#
#         ax[1].set_title('Kspace', fontsize=30)
#         ax[1].imshow(F_rec_Kspaces[:, :, slice], vmin=0, vmax=1, cmap=plt.get_cmap('gray'))
#
#         ax[2].set_title('Image', fontsize=30)
#         ax[2].imshow(refine_Img[:, :, slice], vmin=0, vmax=1, cmap=plt.get_cmap('gray'))
#
#         ax[3].set_title('ZF', fontsize=30)
#         ax[3].imshow(ZF_img[:, :, slice], vmin=0, vmax=1, cmap=plt.get_cmap('gray'))
#
#         ax[4].set_title('GD', fontsize=30)
#         ax[4].imshow(fully_sampled_img[:, :, slice], vmin=0, vmax=1, cmap=plt.get_cmap('gray'))
#
#
#         plt.xticks([]), plt.yticks([])
#         plt.show()
#         clear_output(wait=True)
def plot_imgs(rec_imgs, F_rec_Kspaces, fully_sampled_img, ZF_img):
    # plt.figure()

    slices = [0, 40, 50, 60, 70, 79]

    # for slice in range(rec_imgs.shape[2])
    for slice in slices:
        fig, ax = plt.subplots(1, 4, figsize=(40, 10))
        plt.subplots_adjust(hspace=0, wspace=0)
        ax[0].set_title('Final reconstruction', fontsize=30)
        ax[0].imshow(rec_imgs[:, :, slice], vmin=0, vmax=1, cmap=plt.get_cmap('gray'))

        ax[1].set_title('Kspace reconstruction', fontsize=30)
        ax[1].imshow(F_rec_Kspaces[:, :, slice], vmin=0, vmax=1, cmap=plt.get_cmap('gray'))


        ax[2].set_title('ZF', fontsize=30)
        ax[2].imshow(ZF_img[:, :, slice], vmin=0, vmax=1, cmap=plt.get_cmap('gray'))

        ax[3].set_title('Fully sampled image', fontsize=30)
        ax[3].imshow(fully_sampled_img[:, :, slice], vmin=0, vmax=1, cmap=plt.get_cmap('gray'))


        plt.xticks([]), plt.yticks([])
        plt.show()
        clear_output(wait=True)

def save_imgs(rec_imgs,K_rec,fully_sampled_img,ZF_img):

    # plt.figure()
    slices = [0, 40, 50, 60, 70, 79]

    for i,slice in enumerate(slices):
        # a=Image.fromarray(ZF_img[:, :, slice])
        # a.save("SAVE_path/%s.jpeg" % i)
        # rec=rec_imgs[::-1].T
        # rec=np.expand_dims(rec_imgs[:, :, slice], axis=-1)
        # full_sample=np.expand_dims(fully_sampled_img[:, :, slice], axis=-1)
        # ZF=np.expand_dims(ZF_img[:, :, slice], axis=-1)
        # a= np.concatenate([rec ,full_sample ,ZF])
        # a= np.concatenate([rec_imgs[:, :, slice][::-1].T,fully_sampled_img[:, :, slice][::-1].T,ZF_img[:, :, slice][::-1].T] ,axis=-1)

        rec=rec_imgs[:, :, slice][::-1].T
        gd=fully_sampled_img[:, :, slice][::-1].T
        zf=ZF_img[:, :, slice][::-1].T
        # i_rec=I_rec[:, :, slice][::-1].T
        k_rec=K_rec[:, :, slice][::-1].T

        matplotlib.image.imsave("SAVE_path/%s_rec.png" % i, rec,cmap=plt.get_cmap('gray'))
        matplotlib.image.imsave("SAVE_path/%s_gd.png" % i, gd, cmap=plt.get_cmap('gray'))
        # matplotlib.image.imsave("SAVE_path/%s_I_rec.png" % i, i_rec, cmap=plt.get_cmap('gray'))
        matplotlib.image.imsave("SAVE_path/%s_K_rec.png" % i, k_rec, cmap=plt.get_cmap('gray'))
        matplotlib.image.imsave("SAVE_path/%s_zf.png" % i, zf, cmap=plt.get_cmap('gray'))

        # fig, ax = plt.subplots(1, 4, figsize=(40, 10))
        # plt.subplots_adjust(hspace=0, wspace=0)
        # ax[0].set_title('Final reconstruction', fontsize=30)
        # ax[0].imshow(rec_imgs[:, :, slice], vmin=0, vmax=1, cmap=plt.get_cmap('gray'))
        #
        #
        # ax[1].set_title('Kspace reconstruction', fontsize=30)
        # ax[1].imshow(F_rec_Kspaces[:, :, slice], vmin=0, vmax=1, cmap=plt.get_cmap('gray'))
        #
        #
        # ax[2].set_title('ZF', fontsize=30)
        # ax[2].imshow(ZF_img[:, :, slice], vmin=0, vmax=1, cmap=plt.get_cmap('gray'))
        #
        # ax[3].set_title('Fully sampled image', fontsize=30)
        # ax[3].imshow(fully_sampled_img[:, :, slice], vmin=0, vmax=1, cmap=plt.get_cmap('gray'))


        # plt.xticks([]), plt.yticks([])
        # plt.show()
        # clear_output(wait=True)

