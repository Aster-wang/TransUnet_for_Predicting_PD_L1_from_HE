import argparse
import logging
import os
import random
import sys
import time
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from tqdm import tqdm

from unet import UNet
import deeplabv3plus
from datasets.dataset_synapse import Synapse_dataset
from deeplabv3.deeplabv3_model import *
from dense_aspp.DenseASPP import DenseASPP
from dense_aspp.DenseASPP161 import Model_CFG
from fcn.fcn_model import *
from utils import test_single_volume, pd_toexcel, reset_log
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from attention_unet.AttU_Net import AttU_Net
from DAEFormer.DAEFormer import DAEFormer
from unetpp.unetpp import UnetPlusPlus
from unet_v2.UNet_v2 import UNetV2
from MedT.axialnet import MedT
from UCTransnet.UCTransNet import UCTransNet
from UCTransnet import Config as UCTConfig

parser = argparse.ArgumentParser()
parser.add_argument('--volume_path', type=str,
                    default=r'../../data/npz', help='root dir for validating')
parser.add_argument('--img_size', type=int,
                    default=224, help='input patch size of network input')#transunet=400，DAF=224
parser.add_argument('--is_savenii',default=False,action="store_true", help='whether to save results during inference')
parser.add_argument('--list_dir', type=str,
                    default='./lists/lists_Lung', help='list dir')
parser.add_argument('--num_classes', type=int,
                    default=3, help='output channel of network')
parser.add_argument('--n_skip', type=int,
                    default=3, help='using number of skip-connect, default is num')
parser.add_argument('--vit_name', type=str,
                    default='R50-ViT-B_16', help='select one vit model')
parser.add_argument('--vit_patches_size', type=int,
                    default=16, help='vit_patches_size, default is 16')
args = parser.parse_args()

def inference(args, model, test_save_dir, test_save_path=None):
    db_test = args.Dataset(base_dir=args.volume_path, split="test", list_dir=args.list_dir)

    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=14)
    logging.info("{} test iterations per epoch".format(len(testloader)))
    model.eval()
    metric_list = 0.0
    metric_AP_list=[]
    tps_list=[]
    case_name_list=[]
    for i_batch, sampled_batch in tqdm(enumerate(testloader)):
        #已修改
        image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0]
        metric_i = test_single_volume(image, label, model, classes=args.num_classes, patch_size=[args.img_size, args.img_size],
                                      test_save_path=test_save_path, case=case_name)
        metric_list += np.array(metric_i)
        metric_AP_list.append(np.array(metric_i[0][7:9]))
        logging.info('idx %d case %s mean_dice:%.3f mean_iou:%.3f mean_hd95:%.3f AP:%.3f WP:%.3f NP:%.3f nAP:%.3f pAP:%.3f tps_p:%.3f tps_n:%.3f tps:%.3f'
                     % (i_batch, case_name, metric_i[0][0], metric_i[0][1], metric_i[0][2],
                        metric_i[0][4],metric_i[0][5],metric_i[0][6],metric_i[0][7],metric_i[0][8],metric_i[0][9],metric_i[0][10],metric_i[0][11]))
        tps_list.append(np.array(metric_i[0][-3:]))
        case_name_list.append(case_name)
    # 将tps写如excel文件
    tps_list = np.array(tps_list)
    pd_toexcel(data=[case_name_list,tps_list[:,0], tps_list[:,1], tps_list[:,2]], filename=test_save_dir + '/result.xlsx')
    metric_list = metric_list / len(db_test)
    mean_dice = metric_list[0][0]
    mean_iou = metric_list[0][1]
    mean_hd95 = metric_list[0][2]
    #跟dice系数一样
    mean_f1_score=metric_list[0][3]
    mean_ap = metric_list[0][4]
    mean_wp =metric_list[0][5]
    mean_np = metric_list[0][6]

    metric_AP_list = np.array(metric_AP_list)
    temp= metric_AP_list[:,0]
    temp = temp[temp!=0]
    if len(temp)==0:
        negative_AP=0
    else:
        negative_AP = np.mean(temp)

    temp= metric_AP_list[:,1]
    temp = temp[temp!=0]
    if len(temp)==0:
        positive_AP = 0
    else:
        positive_AP = np.mean(temp)
    logging.info('Testing performance in best val model: '
                 'mean_dice(F1 score) : %.5f mean_iou(Jaccard): %.5f '
                 'mean_hd95 : %.5f mean_f1_score: %.5f mean_ap: %.5f mean_wp: %.5f mean_np: %.5f'
                 'negative_AP: %.5f positive_AP: %.5f' %
                 (mean_dice, mean_iou , mean_hd95,mean_f1_score, mean_ap,mean_wp,mean_np,negative_AP,positive_AP))
    return "Testing Finished!"



if __name__ == "__main__":
    args.z_spacing = 1
    args.is_pretrain = False
    args.Dataset = Synapse_dataset

    # config_vit = CONFIGS_ViT_seg[args.vit_name]
    # config_vit.n_classes = args.num_classes
    # config_vit.n_skip = args.n_skip
    # config_vit.patches.size = (args.vit_patches_size, args.vit_patches_size)
    # if args.vit_name.find('R50') !=-1:
    #     config_vit.patches.grid = (int(args.img_size/args.vit_patches_size), int(args.img_size/args.vit_patches_size))
    # net = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).cuda()
    # 选择模型
    # net = UNet(in_channels=3, num_classes=3, base_c=32).cuda()
    # net = UNetV2(n_classes=3, deep_supervision=False).cuda()
    # net =fcn_resnet101(num_classes=3).cuda()
    # net = DenseASPP(Model_CFG,in_ch=3,out_ch=3).cuda()
    # net = deeplabv3_resnet101(aux=False, num_classes=3).cuda()
    #512x512
    # net = deeplabv3plus.modeling.__dict__['deeplabv3plus_hrnetv2_48'](num_classes=3, output_stride=16,
    #                                                          pretrained_backbone=False).cuda()
    # net = deeplabv3_mobilenetv3_large(num_classes=3).cuda()
    # net = deeplabv3plus.modeling.__dict__['deeplabv3plus_resnet101'](num_classes=3, output_stride=16,
    #                                                                  pretrained_backbone=False).cuda()
    # net=AttU_Net(n_classes=3).cuda()
    # net =lraspp_mobilenet_v3_large(num_classes=3).cuda()
    net=DAEFormer(num_classes=args.num_classes).cuda()
    # net=UnetPlusPlus(num_classes=args.num_classes).cuda()
    # net = MedT().cuda()
    # config_vit = UCTConfig.get_CTranS_config()
    # net = UCTransNet(config_vit, n_channels=3, n_classes=args.num_classes).cuda()
    #手动添加权重
    snapshot = r'./model/size224_epo200_bs16_lr0.00045_DAEFormer_0.6ce-754/epoch_170_0.839.pth'
    net.load_state_dict(torch.load(snapshot))
    weight_name = snapshot.split('/')[-1]

    snapshot_name =weight_name + time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    test_save_dir = './model/' + snapshot.split('/')[-2]
    # log_folder = './test_log'
    # os.makedirs(log_folder, exist_ok=True)
    reset_log(test_save_dir + '/'+snapshot_name+".txt")
    logging.basicConfig(filename=test_save_dir + '/'+snapshot_name+".txt", level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    logging.info(snapshot_name)

    if args.is_savenii:
        # args.test_save_dir = './test_log' #'./predictions'
        test_save_path = os.path.join(test_save_dir, snapshot_name)
        os.makedirs(test_save_path, exist_ok=True)
    else:
        test_save_path = None
    inference(args, net, test_save_dir, test_save_path)


