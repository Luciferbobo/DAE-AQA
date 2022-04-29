# -*- coding: utf-8 -*-


import torch
import logging
import math
import torch.nn as nn

alpha,beta=0.6,0.4

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

def loss_function(sigma, x, mu):
    MSE_loss = nn.MSELoss(reduction='sum')
    rec_loss = alpha/((sigma**2)*MSE_loss(x, mu))
    sup_loss = beta*math.log(sigma**2)
    return rec_loss+sup_loss
