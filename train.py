#!/usr/bin/python
# -*- encoding: utf-8 -*-
from PIL import Image
import numpy as np
from glob import glob
from torch.autograd import Variable
from models.vmamba_Fusion_efficross import VSSM_Fusion
from TaskFusion_dataset import Fusion_dataset
import argparse
import datetime
import time
import logging
import os.path as osp
import os
from logger import setup_logger


from loss import Fusionloss

import torch
from torch.utils.data import DataLoader
import warnings
warnings.filterwarnings('ignore')

def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--use_dsdam', action='store_true', default=False,
                       help='Use DSDAM module in the model')
    parse.add_argument('--dsdam_position', type=str, default='pre',
                       choices=['pre', 'post', 'both'],
                       help='Position of DSDAM: pre (before fusion), post (after fusion), both')
    return parse.parse_args()

def RGB2YCrCb(input_im):
    im_flat = input_im.transpose(1, 3).transpose(
        1, 2).reshape(-1, 3)  # (nhw,c)
    R = im_flat[:, 0]
    G = im_flat[:, 1]
    B = im_flat[:, 2]
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cr = (R - Y) * 0.713 + 0.5
    Cb = (B - Y) * 0.564 + 0.5
    Y = torch.unsqueeze(Y, 1)
    Cr = torch.unsqueeze(Cr, 1)
    Cb = torch.unsqueeze(Cb, 1)
    temp = torch.cat((Y, Cr, Cb), dim=1).cuda()
    out = (
        temp.reshape(
            list(input_im.size())[0],
            list(input_im.size())[2],
            list(input_im.size())[3],
            3,
        )
        .transpose(1, 3)
        .transpose(2, 3)
    )
    return out

def YCrCb2RGB(input_im):
    im_flat = input_im.transpose(1, 3).transpose(1, 2).reshape(-1, 3)
    mat = torch.tensor(
        [[1.0, 1.0, 1.0], [1.403, -0.714, 0.0], [0.0, -0.344, 1.773]]
    ).cuda()
    bias = torch.tensor([0.0 / 255, -0.5, -0.5]).cuda()
    temp = (im_flat + bias).mm(mat).cuda()
    out = (
        temp.reshape(
            list(input_im.size())[0],
            list(input_im.size())[2],
            list(input_im.size())[3],
            3,
        )
        .transpose(1, 3)
        .transpose(2, 3)
    )
    return out

def train_fusion(num=0, logger=None):
    lr_start = 0.0002
    modelpth = 'model_last'
    Method = 'my_cross'
    modelpth = os.path.join(modelpth, Method)
    
    # ========== 消融实验开关：直接在这里修改 ==========
    # use_dsdam: 是否使用DSDAM模块（False=不使用，True=使用）
    # DSDAM 统一放置在每层 DFFM 输入之前(论文规则3), 不再区分 pre/post 位置
    # share_encoder_weights: 双流编码器权重共享开关(论文规则1)
    #   False(默认): IR/VIS 两套独立 encoder 权重
    #   True: IR/VIS 共享同一套 encoder 权重
    use_dsdam = False  # <-- 修改这里切换
    share_encoder_weights = False  # <-- 修改这里切换编码器权重共享
    # ================================================

    fusionmodel = VSSM_Fusion(use_dsdam=use_dsdam, share_encoder_weights=share_encoder_weights)
    print(f"Model: use_dsdam={use_dsdam}, DSDAM位置=每层DFFM输入前(论文规则3), "
          f"share_encoder_weights={share_encoder_weights}")
    
    fusionmodel.cuda()
    fusionmodel.train()
    optimizer = torch.optim.Adam(fusionmodel.parameters(), lr=lr_start)
    train_dataset = Fusion_dataset('train',length=30000)
    print("the training dataset is length:{}".format(train_dataset.length))
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=2,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        drop_last=True,
    )
    train_loader.n_iter = len(train_loader)
    criteria_fusion = Fusionloss()
    
    # 用于记录损失历史
    loss_history = {
        'step': [],
        'loss_total': [],
        'loss_in': [],
        'loss_grad': [],
        'ssim_loss': []
    }

    epoch = 2
    st = glob_st = time.time()
    logger.info('Training Fusion Model start~')
    for epo in range(0, epoch):
        # print('\n| epo #%s begin...' % epo)
        lr_start = 0.0001
        lr_decay = 0.75
        lr_this_epo = lr_start * lr_decay ** (epo - 1)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_this_epo
        for it, (image_vis, image_ir) in enumerate(train_loader):
            try:
                fusionmodel.train()
                image_vis = Variable(image_vis).cuda()
                # image_vis_ycrcb = image_vis[:,0:1:,:,:]
                image_ir = Variable(image_ir).cuda()
                fusion_image = fusionmodel(image_vis, image_ir)

            except TypeError as e:
                print(f"Caught TypeError: {e}")


            ones = torch.ones_like(fusion_image)
            zeros = torch.zeros_like(fusion_image)
            fusion_image = torch.where(fusion_image > ones, ones, fusion_image)
            fusion_image = torch.where(fusion_image < zeros, zeros, fusion_image)
            optimizer.zero_grad()


            # fusion loss
            loss_fusion,  loss_in, ssim_loss, loss_grad= criteria_fusion(
                image_vis=image_vis, image_ir=image_ir, generate_img=
                fusion_image, i=num, labels=None
            )



            loss_total = loss_fusion
            loss_total.backward()
            optimizer.step()
            ed = time.time()
            t_intv, glob_t_intv = ed - st, ed - glob_st
            now_it = train_loader.n_iter * epo + it + 1
            eta = int((train_loader.n_iter * epoch - now_it)
                      * (glob_t_intv / (now_it)))
            eta = str(datetime.timedelta(seconds=eta))
            if now_it % 10 == 0:
                msg = ', '.join(
                    [
                        'step: {it}/{max_it}',
                        'loss_total: {loss_total:.4f}',
                        'loss_in: {loss_in:.4f}',
                        'loss_grad: {loss_grad:.4f}',
                        'ssim_loss: {ssim_loss:.4f}',
                        'eta: {eta}',
                        'time: {time:.4f}',
                    ]
                ).format(
                    it=now_it,
                    max_it=train_loader.n_iter * epoch,
                    loss_total=loss_total.item(),
                    loss_in=loss_in.item(),
                    loss_grad=loss_grad.item(),
                    ssim_loss=ssim_loss.item(),
                    time=t_intv,
                    eta=eta,
                )
                logger.info(msg)
                st = ed
                
                # 记录损失历史
                loss_history['step'].append(now_it)
                loss_history['loss_total'].append(loss_total.item())
                loss_history['loss_in'].append(loss_in.item())
                loss_history['loss_grad'].append(loss_grad.item())
                loss_history['ssim_loss'].append(ssim_loss.item())
                
    fusion_model_file = os.path.join(modelpth, 'fusion_model.pth')
    torch.save(fusionmodel.state_dict(), fusion_model_file)
    logger.info("Fusion Model Save to: {}".format(fusion_model_file))
    logger.info('\n')
    
    # 保存损失历史
    loss_history_file = os.path.join(modelpth, 'loss_history.pth')
    torch.save(loss_history, loss_history_file)
    print(f"Loss history saved to: {loss_history_file}")
    
    # 绘制损失曲线
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Training Loss Curves (use_dsdam={use_dsdam})', fontsize=14)
        
        steps = loss_history['step']
        
        axes[0, 0].plot(steps, loss_history['loss_total'], 'b-', linewidth=1.5)
        axes[0, 0].set_xlabel('Step')
        axes[0, 0].set_ylabel('Total Loss')
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].grid(True)
        
        axes[0, 1].plot(steps, loss_history['loss_in'], 'r-', linewidth=1.5)
        axes[0, 1].set_xlabel('Step')
        axes[0, 1].set_ylabel('Intensity Loss')
        axes[0, 1].set_title('Intensity Preservation Loss')
        axes[0, 1].grid(True)
        
        axes[1, 0].plot(steps, loss_history['loss_grad'], 'g-', linewidth=1.5)
        axes[1, 0].set_xlabel('Step')
        axes[1, 0].set_ylabel('Gradient Loss')
        axes[1, 0].set_title('Gradient Preservation Loss')
        axes[1, 0].grid(True)
        
        axes[1, 1].plot(steps, loss_history['ssim_loss'], 'm-', linewidth=1.5)
        axes[1, 1].set_xlabel('Step')
        axes[1, 1].set_ylabel('SSIM Loss')
        axes[1, 1].set_title('SSIM Loss')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        save_path = os.path.join(modelpth, 'loss_curves.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Loss curves saved to: {save_path}")
        
        # 打印最终损失值
        if len(loss_history['loss_total']) > 0:
            print("\n" + "="*50)
            print("Final Loss Values:")
            print(f"  Total Loss:  {loss_history['loss_total'][-1]:.6f}")
            print(f"  Intensity:   {loss_history['loss_in'][-1]:.6f}")
            print(f"  Gradient:    {loss_history['loss_grad'][-1]:.6f}")
            print(f"  SSIM:        {loss_history['ssim_loss'][-1]:.6f}")
            print("="*50 + "\n")
            
    except ImportError:
        print("matplotlib not installed, skipping loss curve plot")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train with pytorch')
    parser.add_argument('--model_name', '-M', type=str, default='VSSM_Fusion')
    parser.add_argument('--batch_size', '-B', type=int, default=1)
    parser.add_argument('--gpu', '-G', type=int, default=0)
    parser.add_argument('--num_workers', '-j', type=int, default=1)
    args = parser.parse_args()
    logpath='./logs'
    logger = logging.getLogger()
    setup_logger(logpath)
    for i in range(1):
        train_fusion(i, logger)
        print("|{0} Train Fusion Model Sucessfully~!".format(i + 1))
    print("training Done!")
