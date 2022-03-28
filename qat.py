import numpy as np
import os,sys
import argparse
from tqdm import tqdm
from einops import rearrange, repeat

import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from pytorch_nndct.apis import torch_quantizer, dump_xmodel

from ptflops import get_model_complexity_info


sys.path.append(os.path.join(os.path.dirname(sys.argv[0]), "Uformer"))

import scipy.io as sio
from utils.loader import get_validation_data
import utils

from model import UNet,Uformer,Uformer_Cross,Uformer_CatCross

from skimage import img_as_float32, img_as_ubyte
from skimage.metrics import peak_signal_noise_ratio as psnr_loss
from skimage.metrics import structural_similarity as ssim_loss

from pytorch_nndct import nn as nndct_nn
from pytorch_nndct.nn.modules import functional
from pytorch_nndct import QatProcessor


import os
import sys

# add dir
dir_name = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(dir_name,'./auxiliary/'))

import argparse
import options
######### parser ###########
opt = options.Options().init(argparse.ArgumentParser(description='image denoising')).parse_args()
print(opt)

import utils
######### Set GPUs ###########
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
import torch
torch.backends.cudnn.benchmark = True

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(device)
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from natsort import natsorted
import glob
import random
import time
import numpy as np
from einops import rearrange, repeat
import datetime
from pdb import set_trace as stx

from losses import CharbonnierLoss

from tqdm import tqdm 
from warmup_scheduler import GradualWarmupScheduler
from torch.optim.lr_scheduler import StepLR
from timm.utils import NativeScaler

from utils.loader import  get_training_data,get_validation_data


import warnings
warnings.filterwarnings("ignore")


######### Logs dir ###########
log_dir = os.path.join(dir_name,'qat_log', opt.arch+opt.env)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
logname = os.path.join(log_dir, datetime.datetime.now().isoformat()+'.txt') 
print("Now time is : ",datetime.datetime.now().isoformat())
result_dir = os.path.join(log_dir, 'results')
model_dir  = os.path.join(log_dir, 'models')
utils.mkdir(result_dir)
utils.mkdir(model_dir)

# ######### Set Seeds ###########
random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)

######### Model ###########
model_restoration = utils.get_arch(opt)

with open(logname,'a') as f:
    f.write(str(opt)+'\n')
    f.write(str(model_restoration)+'\n')


######### DataParallel ###########
#model_restoration.cuda()

######### Loss ###########
#criterion = CharbonnierLoss().cuda()
criterion = CharbonnierLoss()
######### DataLoader ###########
print('===> Loading datasets')
img_options_train = {'patch_size':opt.train_ps}
train_dataset = get_training_data(opt.train_dir, img_options_train)
train_loader = DataLoader(dataset=train_dataset, batch_size=opt.batch_size, shuffle=True, 
        num_workers=opt.train_workers, pin_memory=True, drop_last=False)

val_dataset = get_validation_data(opt.val_dir)
val_loader = DataLoader(dataset=val_dataset, batch_size=opt.batch_size, shuffle=False, 
        num_workers=opt.eval_workers, pin_memory=False, drop_last=False)

len_trainset = train_dataset.__len__()
len_valset = val_dataset.__len__()
print("Sizeof training set: ", len_trainset,", sizeof validation set: ", len_valset)
######### validation ###########
with torch.no_grad():
    psnr_val_rgb = []
    for ii, data_val in enumerate((val_loader), 0):
        #target = data_val[0].cuda()
        #input_ = data_val[1].cuda()

        target = data_val[0]
        input_ = data_val[1]

        filenames = data_val[2]
        psnr_val_rgb.append(utils.batch_PSNR(input_, target, False).item())
    psnr_val_rgb = sum(psnr_val_rgb)/len_valset
    print('Input & GT (PSNR) -->%.4f dB'%(psnr_val_rgb))



input = torch.randn([opt.batch_size, 3, 256, 256])
#qat_processor = QatProcessor(model_restoration, input, bitwidth=8, device=torch.device('cuda:{}'.format(opt.gpu)))
qat_processor = QatProcessor(model_restoration, input, bitwidth=8, device=torch.device('cpu'))
quantized_model = qat_processor.trainable_model()

######### Optimizer ###########
start_epoch = 1
#if opt.optimizer.lower() == 'adam':
optimizer = optim.Adam(quantized_model.parameters(), lr=opt.lr_initial, betas=(0.9, 0.999),eps=1e-8, weight_decay=opt.weight_decay)
#elif opt.optimizer.lower() == 'adamw':
#        optimizer = optim.AdamW(quantized_model.parameters(), lr=opt.lr_initial, betas=(0.9, 0.999),eps=1e-8, weight_decay=opt.weight_decay)
#else:
#    raise Exception("Error optimizer...")

######### Scheduler ###########
if opt.warmup:
    print("Using warmup and cosine strategy!")
    warmup_epochs = opt.warmup_epochs
    scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.nepoch-warmup_epochs, eta_min=1e-6)
    scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine)
    scheduler.step()
else:
    step = 50
    print("Using StepLR,step={}!".format(step))
    scheduler = StepLR(optimizer, step_size=step, gamma=0.5)
    scheduler.step()

#quantized_model.cuda()


######### train ###########
print('===> Start Epoch {} End Epoch {}'.format(start_epoch,opt.nepoch))
best_psnr = 0
best_epoch = 0
best_iter = 0
eval_now = len(train_loader)//4
print("\nEvaluation after every {} Iterations !!!\n".format(eval_now))

if opt.qat_mode == 'train':
    
    quantized_model.train()

    loss_scaler = NativeScaler()
    torch.cuda.empty_cache()
    for epoch in range(start_epoch, opt.nepoch + 1):
        epoch_start_time = time.time()
        epoch_loss = 0
        train_id = 1

        for i, data in enumerate(train_loader, 0): 
            # zero_grad
            optimizer.zero_grad()

            #target = data[0].cuda()
            #input_ = data[1].cuda()

            target = data[0]
            input_ = data[1]

            if epoch>5:
                target, input_ = utils.MixUp_AUG().aug(target, input_)
            
            #with torch.cuda.amp.autocast():
            restored = quantized_model(input_)
            
            restored = torch.clamp(restored,0,1)  
            loss = criterion(restored, target)
            
            #loss_scaler(
            #        loss, optimizer,parameters=quantized_model.parameters())
            epoch_loss +=loss.item()
            print("debug")
            #### Evaluation ####
            if (i+1)%eval_now==0 and i>0:
                with torch.no_grad():
                    quantized_model.eval()
                    psnr_val_rgb = []
                    for ii, data_val in enumerate((val_loader), 0):
                        #target = data_val[0].cuda()
                        #input_ = data_val[1].cuda()
                        target = data_val[0]
                        input_ = data_val[1]

                        filenames = data_val[2]         
                        
                        #with torch.cuda.amp.autocast():
                        restored = quantized_model(input_)
                        restored = torch.clamp(restored,0,1)  
                        psnr_val_rgb.append(utils.batch_PSNR(restored, target, False).item())
                        
                    psnr_val_rgb = sum(psnr_val_rgb)/len_valset
                    
                    if psnr_val_rgb > best_psnr:
                        best_psnr = psnr_val_rgb
                        best_epoch = epoch
                        best_iter = i 
                        torch.save({'epoch': epoch, 
                                    'state_dict': quantized_model.state_dict(),
                                    'optimizer' : optimizer.state_dict()
                                    }, os.path.join(model_dir,"model_best.pth"))

                    print("[Ep %d it %d\t PSNR SIDD: %.4f\t] ----  [best_Ep_SIDD %d best_it_SIDD %d Best_PSNR_SIDD %.4f] " % (epoch, i, psnr_val_rgb,best_epoch,best_iter,best_psnr))
                    with open(logname,'a') as f:
                        f.write("[Ep %d it %d\t PSNR SIDD: %.4f\t] ----  [best_Ep_SIDD %d best_it_SIDD %d Best_PSNR_SIDD %.4f] " \
                            % (epoch, i, psnr_val_rgb,best_epoch,best_iter,best_psnr)+'\n')
                    quantized_model.train()
                    torch.cuda.empty_cache()
        scheduler.step()
        
        print("------------------------------------------------------------------")
        print("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tLearningRate {:.6f}".format(epoch, time.time()-epoch_start_time,epoch_loss, scheduler.get_lr()[0]))
        print("------------------------------------------------------------------")
        with open(logname,'a') as f:
            f.write("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tLearningRate {:.6f}".format(epoch, time.time()-epoch_start_time,epoch_loss, scheduler.get_lr()[0])+'\n')

        torch.save({'epoch': epoch, 
                    'state_dict': quantized_model.state_dict(),
                    'optimizer' : optimizer.state_dict()
                    }, os.path.join(model_dir,"model_latest.pth"))   

        if epoch%opt.checkpoint == 0:
            torch.save({'epoch': epoch, 
                        'state_dict': quantized_model.state_dict(),
                        'optimizer' : optimizer.state_dict()
                        }, os.path.join(model_dir,"model_epoch_{}.pth".format(epoch))) 
    print("Now time is : ",datetime.datetime.now().isoformat())

    deployable_model = qat_processor.to_deployable(quantized_model,opt.save_dir)


elif opt.qat_mode == 'deploy':
    # Step 3: Export xmodel from deployable model.
    deployable_model = qat_processor.deployable_model(
        opt.save_dir, used_for_xmodel=True)
    # Must forward deployable model at least 1 iteration with batch_size=1
    for i, data in enumerate(train_loader, 0):
      deployable_model(data[1].cuda())
    qat_processor.export_xmodel(opt.save_dir)










































