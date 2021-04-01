# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   Description :   weight entropy on image level
   Author :        Ymc
   date：          2020/10/9
-------------------------------------------------
"""
import os
import os.path as osp
import sys

import numpy as np
import torch
import torch.optim as optim
from torch import nn
from torch.backends import cudnn
from torch.nn import functional as F
from torch.utils import data
from tqdm import tqdm

sys.path.append(os.path.abspath('.'))
from utils import project_root
from utils.func import get_logger, print_losses, adjust_learning_rate, com_entropy_term2, com_term1, normalization
from dataset.cityscapes import CityscapesDataSet
from model.deeplabv2_AdaptSegNet import Res_Deeplab


class Configuration(object):
    # project name
    ABSTRACT = 'uncertain self training with dropout, layer4=0.1, pre-train model is 41'
    DA_METHOD = os.path.basename(__file__).split('.')[0]

    CUDA_VISIBLE_DEVICES = '1'

    M = 4
    T = 16

    # target dataset
    SOURCE = 'GTA'
    TARGET = 'Cityscapes'
    NUM_WORKERS = 8
    DATA_LIST_TARGET = str(project_root / 'dataset/cityscapes_list/{}.txt')
    DATA_DIRECTORY_TARGET = str('/home/ouyangjinpeng/datasets/city')
    NUM_CLASSES = 19
    EXP_NAME = ''
    EXP_ROOT = project_root / 'new_experiments'
    EXP_ROOT_SNAPSHOT = osp.join(EXP_ROOT, 'snapshots')
    EXP_ROOT_BAK = osp.join(EXP_ROOT, 'bak')
    EXP_ROOT_LOGS = osp.join(EXP_ROOT, 'logs')
    GPU_ID = 0

    # train
    SET_TARGET = 'train'
    VAL = 'val'
    BATCH_SIZE_TARGET = 1
    IGNORE_LABEL = 255
    INPUT_SIZE_TARGET = (1024, 512)
    INFO_SOURCE = ''
    INFO_TARGET = str(project_root / 'dataset/cityscapes_list/info.json')
    MODEL = 'DeepLabv2'
    MULTI_LEVEL = False
    RESTORE_FROM = r'/data2/ouyangjinpeng/project/Finalv4/saved_models/ingore0_9.pth'
    IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)
    LEARNING_RATE = 2.5e-4
    MOMENTUM = 0.9
    WEIGHT_DECAY = 0.0005
    POWER = 0.9
    # MinEnt params
    LAMBDA_ENT_MAIN = 0.001
    MAX_ITERS = 250000
    EARLY_STOP = 120000
    SAVE_PRED_EVERY = 300
    SNAPSHOT_DIR = ''
    RANDOM_SEED = 1234
    LOGDIR = ''
    TENSORBOARD_LOGDIR = ''
    TENSORBOARD_VIZRATE = 100
    SEED = False

    # args
    TENSORBOARD = False
    VIZ_EVERY_ITER = None

    # Test
    T_MODE = 'best'
    T_MODEL = ('DeepLabv2',)
    T_MODEL_WEIGHT = (1.0,)
    T_MULTI_LEVEL = (False,)
    T_IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)
    T_RESTORE_FROM = ['', ]
    T_SNAPSHOT_DIR = ['', ]
    T_SNAPSHOT_STEP = 300
    T_SNAPSHOT_MAXITER = 120000
    T_SET_TARGET = 'val'
    T_BATCH_SIZE_TARGET = 1
    T_INPUT_SIZE_TARGET = (1024, 512)
    T_OUTPUT_SIZE_TARGET = (2048, 1024)
    T_INFO_TARGET = str(project_root / 'dataset/cityscapes_list/info.json')
    T_WAIT_MODEL = True


def main():
    cfg = Configuration()
    if cfg.CUDA_VISIBLE_DEVICES is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = cfg.CUDA_VISIBLE_DEVICES
    if cfg.EXP_NAME == '':
        cfg.EXP_NAME = f'{cfg.SOURCE}2{cfg.TARGET}_{cfg.MODEL}_{cfg.DA_METHOD}'
    # auto-generate snapshot path if not specified
    if cfg.SNAPSHOT_DIR == '':
        cfg.SNAPSHOT_DIR = osp.join(cfg.EXP_ROOT_SNAPSHOT, cfg.EXP_NAME)
        os.makedirs(cfg.SNAPSHOT_DIR, exist_ok=True)
    # logging
    if cfg.LOGDIR == '':
        cfg.LOGDIR = cfg.EXP_ROOT_LOGS
    os.makedirs(cfg.LOGDIR, exist_ok=True)
    logger = get_logger(cfg)
    logger.info(str(cfg.ABSTRACT))
    print(str(cfg.ABSTRACT))
    print(str(cfg.DA_METHOD))

    assert osp.exists(cfg.RESTORE_FROM), f'Missing init model {cfg.RESTORE_FROM}'
    if cfg.MODEL == 'DeepLabv2':
        if cfg.MULTI_LEVEL:
            raise TypeError('aux is not')
        else:
            model = Res_Deeplab(num_classes=cfg.NUM_CLASSES)
            saved_state_dict = torch.load(cfg.RESTORE_FROM)
            model.load_state_dict(saved_state_dict)

            model_keep = Res_Deeplab(num_classes=cfg.NUM_CLASSES)
            model_keep.load_state_dict(saved_state_dict)
    else:
        raise NotImplementedError(f"Not yet supported {cfg.MODEL}")

    target_dataset = CityscapesDataSet(root=cfg.DATA_DIRECTORY_TARGET,
                                       list_path=cfg.DATA_LIST_TARGET,
                                       set=cfg.SET_TARGET,
                                       info_path=cfg.INFO_TARGET,
                                       max_iters=cfg.MAX_ITERS * cfg.BATCH_SIZE_TARGET,
                                       crop_size=cfg.INPUT_SIZE_TARGET,
                                       mean=cfg.IMG_MEAN)
    target_loader = data.DataLoader(target_dataset,
                                    batch_size=cfg.BATCH_SIZE_TARGET,
                                    num_workers=cfg.NUM_WORKERS,
                                    shuffle=True,
                                    pin_memory=False)
    # UDA TRAINING
    input_size_target = cfg.INPUT_SIZE_TARGET
    device = cfg.GPU_ID
    model.train()
    model.to(device)

    model_keep.eval()
    model_keep.to(device)

    cudnn.benchmark = True
    cudnn.enabled = True

    optimizer = optim.SGD(model.optim_parameters(cfg.LEARNING_RATE),
                          lr=cfg.LEARNING_RATE,
                          momentum=cfg.MOMENTUM,
                          weight_decay=cfg.WEIGHT_DECAY)

    interp_target = nn.Upsample(size=(input_size_target[1], input_size_target[0]), mode='bilinear',
                                align_corners=True)
    cross_entropy_loss = nn.CrossEntropyLoss(reduction='none', ignore_index=255)
    target_loader_iter = enumerate(target_loader)
    for i_iter in tqdm(range(cfg.EARLY_STOP)):

        optimizer.zero_grad()
        adjust_learning_rate(optimizer, i_iter, cfg)

        _, batch = target_loader_iter.__next__()
        images, _, _, image_name = batch
        images = images.cuda()

        pred_trg_main = model(images)
        pred_trg_main = interp_target(pred_trg_main)
        # pred_prob_trg_main = F.softmax(pred_trg_main)

        uncertain_list = list()
        with torch.no_grad():

            GT = model_keep(images)
            GT = interp_target(GT)
            GT = F.softmax(GT)
            GT = torch.argmax(GT, dim=1)

            for _ in range(cfg.T):
                U = model_keep(images, drop_out=True)
                U = interp_target(U)
                U = F.softmax(U)
                uncertain_list.append(U)
            uncertain_feature = torch.cat(uncertain_list, dim=0)

        B = com_term1(uncertain_feature) + com_entropy_term2(uncertain_feature)
        B = normalization(B)
        B = -torch.log2(B + 1e-5)

        seg_loss = cross_entropy_loss(pred_trg_main, GT) * B

        loss = torch.mean(seg_loss)
        loss.backward()
        optimizer.step()

        current_losses = {'loss_ent_main': loss}

        print_losses(current_losses, i_iter, logger)

        is_save_iter = int(cfg.SAVE_PRED_EVERY // cfg.BATCH_SIZE_TARGET)

        if i_iter % is_save_iter == 0 and i_iter != 0:
            torch.save(model.state_dict(),
                       osp.join(cfg.SNAPSHOT_DIR, f'model_{i_iter}.pth'))
            if i_iter >= cfg.EARLY_STOP - 1:
                break
        sys.stdout.flush()


if __name__ == '__main__':
    main()
