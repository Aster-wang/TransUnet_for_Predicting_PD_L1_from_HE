import argparse
import logging
import os
import random
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.ndimage import zoom
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.dataset_synapse import change
from utils import DiceLoss, get_exp, test_single_volume, reset_log, Tversky_Loss
from torchvision import transforms
import matplotlib.pyplot as plt

from skimage.metrics import structural_similarity as ssim

def worker_init_fn(worker_id):
    random.seed(1234 + worker_id)

def trainer_synapse(args, model, snapshot_path):
    # from datasets.dataset_synapse_boundary import Synapse_dataset, RandomGenerator
    from datasets.dataset_synapse import Synapse_dataset, RandomGenerator
    reset_log(snapshot_path + "/log.txt")
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    nw = args.nw
    print('Using %g dataloader workers' % nw)

    #训练数据加载
    db_train = Synapse_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="train",
                               transform=transforms.Compose(
                                   [RandomGenerator(output_size=[args.img_size, args.img_size])]))
    print("The length of train set is: {}".format(len(db_train)))

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=nw,
                             worker_init_fn=worker_init_fn,drop_last=True)

    #验证数据加载
    db_val = Synapse_dataset(base_dir=args.root_path, split="val", list_dir=args.list_dir,
                             change=transforms.Compose(
                                 [change(output_size=[args.img_size, args.img_size])])
                             )
    print("The length of val set is: {}".format(len(db_val)))
    otherloader = DataLoader(db_val, batch_size=batch_size, shuffle=True, num_workers=nw,drop_last=True)



    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)
    # optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    optimizer = optim.Adam(params=model.parameters(),lr=base_lr)
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    val_loss_list=[]
    train_loss_list=[]
    val_dice_list = []
    train_dice_list = []
    train_loss_ce_list = []
    train_loss_dc_list = []
    best_acc = 0.7
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        # 开始训练
        model.train()
        train_loss=0.0
        train_dice=0.0
        train_dc_loss = 0.0
        train_ce_loss = 0.0
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()

            outputs = model(image_batch)

            loss_ce = ce_loss(outputs, label_batch[:].long())
            loss_dice = dice_loss(outputs, label_batch, softmax=True)
            train_dice += (1-loss_dice).cpu().detach().numpy()

            loss = 0.6*loss_ce+0.4*loss_dice

            optimizer.zero_grad()
            loss.backward()
            train_loss += loss
            optimizer.step()
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_
            iter_num = iter_num + 1

            train_ce_loss += loss_ce
            train_dc_loss += loss_dice

        mean_train_dice=train_dice/len(trainloader)
        train_dice_list.append(mean_train_dice)

        train_loss=train_loss.cpu().detach().numpy()
        mean_train_loss=train_loss/len(trainloader)
        train_loss_list.append(mean_train_loss)
        '''celoss diceloss 变化'''
        train_ce_loss=train_ce_loss.cpu().detach().numpy()
        mean_train_ce_loss=train_ce_loss/len(trainloader)
        train_loss_ce_list.append(mean_train_ce_loss)

        train_dc_loss=train_dc_loss.cpu().detach().numpy()
        mean_train_dc_loss=train_dc_loss/len(trainloader)
        train_loss_dc_list.append(mean_train_dc_loss)

        # 计算loss (只使用dice作为验证指标)
        val_loss = 0.0
        dice_all=0.0
        model.eval()
        for i_batch, sampled_batch in enumerate(otherloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
            with torch.no_grad():
                outputs = model(image_batch)
                loss_ce = ce_loss(outputs, label_batch[:].long())
                loss_dice = dice_loss(outputs, label_batch, softmax=True)#, argmax=True
                val_loss += (loss_ce * 0.5 + loss_dice * 0.5).cpu()
                val_dice = (1-loss_dice).cpu().detach().numpy()

                dice_all+=val_dice

        mean_val_dice=dice_all/len(otherloader)
        val_dice_list.append(mean_val_dice)
        val_loss = val_loss.detach().numpy()

        val_loss_list.append(val_loss/len(otherloader))
        if mean_val_dice >best_acc:
            best_acc = mean_val_dice
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num+1)+'_'+str(np.round(mean_val_dice,3))+ '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
        save_weight_epoch = args.max_epochs - 2
        if epoch_num > save_weight_epoch:
            save_mode_path = os.path.join(snapshot_path, 'z_epoch_' + str(epoch_num + 1) + '_' + str(np.round(mean_val_dice, 3)) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
        # logging.info('epoch %d : train_loss : %f, train_dice : %f, val_dice: %f, lr: %f ' % (
        #     epoch_num, mean_train_loss.item(), mean_train_dice.item(), mean_val_dice.item(), lr_))
        logging.info('epoch %d : train_loss : %f, train_ce_loss : %f, train_dc_loss : %f, train_dice : %f, val_dice: %f, lr: %f ' % (
            epoch_num, mean_train_loss.item(), mean_train_ce_loss.item(), mean_train_dc_loss.item(), mean_train_dice.item(), mean_val_dice.item(), lr_))

    print(val_dice_list)
    print(train_dice_list)
    plt.plot(val_loss_list,label='val_loss')
    plt.plot(train_loss_list,label='train_loss')
    plt.plot(train_loss_ce_list,label='train_ce_loss')
    plt.plot(train_loss_dc_list,label='train_dc_loss')
    plt.title('loss')
    plt.legend()
    plt.savefig(snapshot_path + '/loss.png')
    plt.clf()
    plt.plot(val_dice_list, label='val')
    plt.plot(train_dice_list, label='train')
    plt.title('dice')
    plt.legend()
    plt.savefig(snapshot_path + '/dice.png')
    return "Training Finished!"