import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import os
import sys
from scipy import stats
from tqdm import tqdm
import argparse
import numpy as np
import glob
from PIL import Image
import pickle as pkl
import logging
from models import InceptionI3d,DAE
from dataloader import load_image_train,load_image,VideoDataset,get_dataloaders
from config import get_parser
from util import get_logger,log_and_print,loss_function

sys.path.append('../')
torch.backends.cudnn.enabled = True
i3d_pretrained_path = './ckpts/rgb_i3d_pretrained.pt'
feature_dim = 1024


if __name__ == '__main__':

    args = get_parser().parse_known_args()[0]

    if not os.path.exists('./exp'):
        os.mkdir('./exp')
    if not os.path.exists('./ckpts'):
        os.mkdir('./ckpts')

    base_logger = get_logger(f'exp/DAE.log', args.log_info)
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    i3d = InceptionI3d().cuda()
    i3d.load_state_dict(torch.load(i3d_pretrained_path))

    dae = DAE().cuda()
    dataloaders = get_dataloaders(args)

    optimizer = torch.optim.Adam([*i3d.parameters()] + [*dae.parameters()],
                                 lr=args.lr, weight_decay=args.weight_decay)

    epoch_best = 0
    rho_best = 0.9
    for epoch in range(args.num_epochs):
        log_and_print(base_logger, f'Epoch: {epoch}')

        for split in ['train', 'test']:
            true_scores = []
            pred_scores = []
            sigma = []

            if split == 'train':
                i3d.eval()
                dae.train()
                torch.set_grad_enabled(True)
            else:
                i3d.eval()
                dae.eval()
                torch.set_grad_enabled(False)

            for data in tqdm(dataloaders[split]):
                true_scores.extend(data['final_score'].numpy())
                videos = data['video'].cuda()
                videos.transpose_(1, 2)  # N, C, T, H, W

                batch_size, C, frames, H, W = videos.shape
                clip_feats = torch.empty(batch_size, 10, feature_dim).cuda()
                for i in range(9):
                    clip_feats[:, i] = i3d(videos[:, :, 10 * i:10 * i + 16, :, :]).squeeze(2)
                clip_feats[:, 9] = i3d(videos[:, :, -16:, :, :]).squeeze(2)

                preds, mu, sigmas = dae(clip_feats.mean(1))
                preds = preds.view(-1)
                sigmas = sigmas.view(-1)
                mu = mu.view(-1)               
                pred_scores.extend([i.item() for i in preds])
                sigma.extend([i.item() for i in sigmas])

                if split == 'train':
                    loss = loss_function(sigma, data['final_score'].float().cuda(), mu)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            rho, p = stats.spearmanr(pred_scores, true_scores)

            log_and_print(base_logger, f'{split} correlation: {rho}')

        if rho > rho_best:
            rho_best = rho
            epoch_best = epoch
            log_and_print(base_logger, '##### New best correlation #####')
            path = 'ckpts/' + str(rho) + '.pt'
            torch.save({'epoch': epoch,
                            'i3d': i3d.state_dict(),
                            'dae': dae.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'rho_best': rho_best}, path)
