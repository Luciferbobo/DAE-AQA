# -*- coding: utf-8 -*-


import torch
import logging
import math
import torch.nn as nn
import numpy as np

alpha,beta=torch.Tensor([0.6]), torch.Tensor([0.4])

def get_logger(filepath, log_info):
    logger = logging.getLogger(filepath)
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(filepath)
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)
    logger.info('-' * 30 + log_info + '-' * 30)
    return logger


def log_and_print(logger, msg):
    logger.info(msg)
    print(msg)

def loss_function_v2(sigma, x, mu):
    MSE_loss = nn.MSELoss(reduction='sum')
    rec_loss = alpha/(sigma)*MSE_loss(x, mu)
    sup_loss = beta*torch.log(sigma)
    return rec_loss+sup_loss

def loss_function(recon_x, x, mu):
    MSE_loss = nn.MSELoss(reduction='sum')
    reconstruction_loss = MSE_loss(recon_x, x)+MSE_loss(x, mu)
    return reconstruction_loss

