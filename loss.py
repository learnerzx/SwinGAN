import torch
from torch.autograd import Function
import torch.nn as nn
import pickle
from focal_frequency_loss import FocalFrequencyLoss as FFL
import torch
from scipy.io import loadmat, savemat
import cv2

class netLoss():

    def __init__(self, args, masked_kspace_flag=True):
        self.args = args

        mask_path = args.mask_path


        if args.mask_type == 'radial' and args.sampling_percentage == 30:
            with open(mask_path, 'rb') as pickle_file:
                masks = pickle.load(pickle_file)
                self.mask = torch.tensor(masks == 1, device=self.args.device)
            self.masked_kspace_flag = masked_kspace_flag
        elif args.mask_type == 'radial' and args.sampling_percentage == 50:
            mask_shift = cv2.imread(r'E:\code\code_backup\Masks\radial\radial_50.tif', 0) / 255
            # mask = scipy.ifft(mask_shift)
            self.mask = torch.tensor(mask_shift == 1, device=self.args.device)
            self.masked_kspace_flag = masked_kspace_flag

        elif args.mask_type == 'random':
            with open(mask_path, 'rb') as pickle_file:
                masks_dictionary = pickle.load(pickle_file)
                self.masks = masks_dictionary['mask1']
                self.mask = torch.tensor(self.masks == 1, device=self.args.device)
                self.masked_kspace_flag = masked_kspace_flag
        else:
            masks_dictionary = loadmat(mask_path)
            self.masked_kspace_flag = masked_kspace_flag
            try:
                self.mask = torch.tensor(masks_dictionary['Umask'] == 1, device=self.args.device)
            except:
                try:
                    self.mask = torch.tensor(masks_dictionary['maskRS2'] == 1, device=self.args.device)
                except:
                    self.mask = torch.tensor(masks_dictionary['population_matrix'] == 1, device=self.args.device)

        # self.mask = torch.tensor(masks_dictionary['Umask']==1, device=self.args.device)
        self.maskNot = self.mask == 0

        # with open(mask_path, 'rb') as pickle_file:
        #     masks = pickle.load(pickle_file)
        # self.masked_kspace_flag = masked_kspace_flag
        # self.mask = torch.tensor(masks['mask1']==1, device=self.args.device)
        # self.maskNot = self.mask == 0


        self.ImL2_weights = args.loss_weights[0]
        self.ImL1_weights = args.loss_weights[1]
        self.KspaceL2_weights = args.loss_weights[2]
        self.AdverLoss_weight = args.loss_weights[3]
        self.FFLLoss_weight =args.loss_weights[4]
        self.ImL2Loss = nn.MSELoss()
        self.ImL1Loss = nn.SmoothL1Loss()

        self.AdverLoss = nn.BCEWithLogitsLoss()

        if self.masked_kspace_flag:
            self.KspaceL2Loss = nn.MSELoss(reduction='sum')
        else:
            self.KspaceL2Loss = nn.MSELoss()

    def img_space_loss(self,pred_Im,tar_Im):
        return self.ImL1Loss(pred_Im, tar_Im),self.ImL2Loss(pred_Im, tar_Im)

    def k_space_loss(self,pred_K,tar_K):
        if self.masked_kspace_flag:
            return self.KspaceL2Loss(pred_K, tar_K)/(torch.sum(self.maskNot)*tar_K.max())
        else:
            return self.KspaceL2Loss(pred_K, tar_K)

    def gen_adver_loss(self,D_fake):
        real_ = torch.tensor(1.0).expand_as(D_fake).to(self.args.device)
        return self.AdverLoss(D_fake, real_)

    def disc_adver_loss(self, D_real, D_fake):
        real_ = torch.tensor(1.0).expand_as(D_real).to(self.args.device)
        fake_ = torch.tensor(0.0).expand_as(D_fake).to(self.args.device)
        real_loss = self.AdverLoss(D_real,real_)
        fake_loss = self.AdverLoss(D_fake,fake_)
        return real_loss,fake_loss

    def calc_gen_loss(self, pred_Im, pred_K, tar_Im, tar_K,D_fake=None):
        ImL1,ImL2 = self.img_space_loss(pred_Im, tar_Im)

        KspaceL2 = self.k_space_loss(pred_K, tar_K)

        ffl = FFL()
        fflLoss=self.FFLLoss=ffl(pred_K,tar_K)

        if D_fake is not None:
            advLoss = self.gen_adver_loss(D_fake)
        else:
            advLoss = 0
        fullLoss = self.ImL2_weights*ImL2 + self.ImL1_weights*ImL1 + self.KspaceL2_weights*KspaceL2 + self.AdverLoss_weight*advLoss + self.FFLLoss_weight*fflLoss
        return fullLoss, ImL2, ImL1, KspaceL2, advLoss, fflLoss

    def calc_disc_loss(self,D_real,D_fake):
        real_loss,fake_loss = self.disc_adver_loss(D_real,D_fake)
        return real_loss,fake_loss, 0.5*(real_loss + fake_loss)

def set_grad(network,requires_grad):
        for param in network.parameters():
            param.requires_grad = requires_grad
# class netLoss(Function):
#     """Dice coeff for individual examples"""
#
#     def forward(self, input, target):
#         self.save_for_backward(input, target)
#         eps = 0.0001
#         self.inter = torch.dot(input.view(-1), target.view(-1))
#         self.union = torch.sum(input) + torch.sum(target) + eps
#
#         t = (2 * self.inter.float() + eps) / self.union.float()
#         return t
#
#     # This function has only a single output, so it gets only one gradient
#     def backward(self, grad_output):
#
#         input, target = self.saved_variables
#         grad_input = grad_target = None
#
#         if self.needs_input_grad[0]:
#             grad_input = grad_output * 2 * (target * self.union - self.inter) \
#                          / (self.union * self.union)
#         if self.needs_input_grad[1]:
#             grad_target = None
#
#         return grad_input, grad_target
#
#
# def dice_coeff(input, target):
#     """Dice coeff for batches"""
#     if input.is_cuda:
#         s = torch.FloatTensor(1).cuda().zero_()
#     else:
#         s = torch.FloatTensor(1).zero_()
#
#     for i, c in enumerate(zip(input, target)):
#         s = s + netLoss().forward(c[0], c[1])
#
#     return s / (i + 1)
