import argparse
import logging
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn

import deeplabv3plus
from deeplabv3.deeplabv3_model import deeplabv3_mobilenetv3_large,deeplabv3_resnet50,deeplabv3_resnet101
from dense_aspp.DenseASPP import DenseASPP, get_parameter_number
from dense_aspp.DenseASPP161 import Model_CFG
from fcn.fcn_model import fcn_resnet50,fcn_resnet101
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from unet import UNet
from attention_unet.AttU_Net import AttU_Net
from trainer import trainer_synapse
from DAEFormer.DAEFormer import DAEFormer
from unetpp.unetpp import UnetPlusPlus
from unet_v2.UNet_v2 import UNetV2
from MedT.axialnet import MedT
from UCTransnet.UCTransNet import UCTransNet
from UCTransnet import Config as UCTConfig
from thop import profile

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default=r'../../data/npz', help='root dir for data')
#F:/Dex/data/PD-L1-npz/4_times_croped/npz  #../../data/npz  #F:/Dex/data/PD-L1-8croped-npz  #../../data/8croped-npz  #../../data/Larynx_npz
parser.add_argument('--list_dir', type=str,
                    default='./lists/lists_Lung', help='list dir')
parser.add_argument('--base_lr', type=float,  default=0.008,
                    help='segmentation network learning rate')
parser.add_argument('--nw', type=int,  default=14,
                    help='number of workers')#min([os.cpu_count(), batch_size if batch_size > 1 else 0, 14])
parser.add_argument('--max_epochs', type=int,
                    default=200, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,
                    default=16, help='batch_size per gpu')
parser.add_argument('--img_size', type=int,
                    default=224, help='input patch size of network input')#transunet=400，DAF=224
parser.add_argument('--seed', type=int,
                    default=1234, help='random seed')

parser.add_argument('--num_classes', type=int,
                    default=3, help='output channel of network')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--n_skip', type=int,
                    default=3, help='using number of skip-connect, default is num')
parser.add_argument('--vit_name', type=str,
                    default='R50-ViT-B_16', help='select one vit model')
parser.add_argument('--vit_patches_size', type=int,
                    default=16, help='vit_patches_size, default is 16')
args = parser.parse_args()


if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    os.environ['PYTHONHASHSEED'] = str(args.seed)  # 设置Python哈希种子，为了禁止hash随机化，使得实验可复现
    torch.cuda.manual_seed_all(args.seed)  # if you are using multi-GPU.   torch的GPU随机性，为所有GPU设置随机种子

    # torch.use_deterministic_algorithms(True)#如果结果不可复现，他会给你报错。
    # os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

    args.is_pretrain = True
    args.exp = 'size' + str(args.img_size)
    snapshot_path = "./model/{}".format(args.exp)
    snapshot_path = snapshot_path + '_vitpatch' + str(args.vit_patches_size) if args.vit_patches_size!=16 else snapshot_path
    snapshot_path = snapshot_path+'_'+str(args.max_iterations)[0:2]+'k' if args.max_iterations != 30000 else snapshot_path
    snapshot_path = snapshot_path + '_epo' +str(args.max_epochs) if args.max_epochs != 30 else snapshot_path
    snapshot_path = snapshot_path+'_bs'+str(args.batch_size)
    snapshot_path = snapshot_path + '_lr' + str(args.base_lr)
    snapshot_path = snapshot_path + '_s'+str(args.seed) if args.seed!=1234 else snapshot_path
    snapshot_path = snapshot_path + '_UnetV2_0.6ce'#_deeplabv3+hrnet3  unet_1   deeplabv3_mobilenetv3_1  DenseASPP_2
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    # config_vit = CONFIGS_ViT_seg[args.vit_name]
    # config_vit.n_classes = args.num_classes
    # config_vit.n_skip = args.n_skip
    # if args.vit_name.find('R50') != -1: #如果包含，则返回第一次出现该字符串的索引；反之，则返回 -1。
    #     config_vit.patches.grid = (int(args.img_size / args.vit_patches_size), int(args.img_size / args.vit_patches_size))
    # net = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes)
    # net.load_from(weights=np.load(config_vit.pretrained_path))

    # net = UNet(in_channels=3, num_classes=3, base_c=32)
    # net = UNetV2(n_classes=3, deep_supervision=False, pretrained_path='../checkpoint/pvt_v2_b2.pth')
    # net = fcn_resnet101(num_classes=3)
    # net = DenseASPP(Model_CFG,in_ch=3,out_ch=3)
    net = deeplabv3_mobilenetv3_large(num_classes=3)
    # net = deeplabv3plus.modeling.__dict__['deeplabv3plus_hrnetv2_48'](num_classes=3, output_stride=16,
    #                                                          pretrained_backbone=False)
    # net = deeplabv3plus.modeling.__dict__['deeplabv3plus_resnet101'](num_classes=3, output_stride=16,
    #                                                           pretrained_backbone=False)
    # net = AttU_Net(n_classes=3)
    # net =deeplabv3_resnet101(num_classes=3)
    # net =lraspp_mobilenet_v3_large(num_classes=3)
    # net=DAEFormer(num_classes=args.num_classes)
    # net=UnetPlusPlus(num_classes=args.num_classes)
    # net = MedT()
    # config_vit = UCTConfig.get_CTranS_config()
    # net = UCTransNet(config_vit, n_channels=3, n_classes=args.num_classes)
    #
    # weight_file = torch.load(r'D:\Dex\workspace\deep-learning-for-image-processing-master\pytorch_segmentation\TransUNet-ori\test_log\EADFormer\第二次训练，SGD\TU_pretrain_R50-ViT-B_16_skip3_epo150_bs8_lr0.05_224\epoch_115_0.883.pth')
    # net.load_state_dict(weight_file, strict=False)

    # net.load_state_dict(torch.load('model/TU_Synapse400/TU_pretrain_R50-ViT-B_16_skip3_epo50_bs2_lr0.005_400/epoch_19_0.927.pth'))
    print(get_parameter_number(net))
    # flops, params = profile(net, inputs=(torch.randn(1,3,224,224),))
    # print('flops:{}'.format(flops))
    # print('params:{}'.format(params))

    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    net.cuda()
    # device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    # print(device)
    # net.to(device)
    trainer = trainer_synapse
    trainer(args, net, snapshot_path)
