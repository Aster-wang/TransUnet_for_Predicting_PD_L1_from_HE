
import math
import numpy as np
import pandas as pd
import torch
from PIL import Image
from medpy import metric
from scipy.ndimage import zoom
import torch.nn as nn

class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False, argmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        if argmax:
            inputs = torch.argmax(inputs, dim=1)
            inputs = self._one_hot_encoder(inputs)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes

class Tversky_Loss(nn.Module):
    def __init__(self, n_classes, alpha=0.7):
        super(Tversky_Loss, self).__init__()
        self.n_classes = n_classes
        self.alpha = alpha
        self.beta = 1 - alpha
    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _tversky_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        tp = torch.sum(score * target)
        fp = torch.sum((1-target)*score)
        fn = torch.sum((1-score)*target)
        tversky = (tp + smooth)/(tp + self.alpha*fp + self.beta*fn + smooth)
        loss = 1 - tversky
        return loss


    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._tversky_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes

#权重无梯度
def get_exp(inputs,target,a=2):
    inputs = inputs.cpu().detach().numpy()
    target = target.cpu().detach().numpy()
    inputs = np.squeeze(np.argmax(inputs,1))
    target = np.squeeze(target)
    assert inputs.shape == target.shape, 'predict {} & target {} shape do not match'.format(inputs.shape,
                                                                                              target.shape)

    loss = np.mean((4 * (np.exp(a * np.fabs(inputs - target)) - 1) / (np.exp(4 * a) - 1)) + 1)
    return loss
#权重无梯度
def get_log(inputs,target,b=10):
    inputs = inputs.cpu().detach().numpy()
    target = target.cpu().detach().numpy()
    inputs = np.squeeze(np.argmax(inputs,1))
    target = np.squeeze(target)

    assert inputs.shape == target.shape, 'predict {} & target {} shape do not match'.format(inputs.shape,
                                                                                              target.shape)

    loss = np.mean(4 * np.log(b * np.fabs(inputs - target) + 1) / np.log(4 * b + 1) + 1)
    return loss

class Class_exp_Loss(nn.Module):
    def __init__(self):
        super(Class_exp_Loss, self).__init__()

    def forward(self, inputs, target,a=2):

        inputs = torch.argmax(inputs, dim=1).squeeze(0)
        target = target.squeeze(0)

        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())

        loss = torch.mean((4*(torch.exp(a*torch.abs(inputs-target))-1)/(torch.exp(4*a)-1))+1)
        return loss

class Class_log_Loss(nn.Module):
    def __init__(self):
        super(Class_log_Loss, self).__init__()

    def forward(self, inputs, target,b=10):

        inputs = torch.argmax(inputs, dim=1).squeeze(0)
        target = target.squeeze(0)

        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())

        loss = np.mean(4*torch.log(b*torch.abs(inputs-target)+1)/np.log(4*b+1)+1)
        return loss

class Class_class_Loss(nn.Module):
    def __init__(self):
        super(Class_class_Loss, self).__init__()

    def forward(self, inputs, target):

        inputs = torch.argmax(inputs, dim=1).squeeze(0)
        target = target.squeeze(0)

        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        loss = torch.mean(torch.abs(torch.sub(inputs,target)))
        return loss

class BoundaryLoss(nn.Module):
    def __init__(self):
        super(BoundaryLoss, self).__init__()

    def get_boundary(self, a):
        obj_pix = torch.ones_like(a,device='cuda:0')
        result = torch.zeros_like(a,device='cuda:0')

        a = torch.where(a > 1, obj_pix, a)
        # （1，0）
        b = a[:, 1:]
        c = torch.zeros(a.shape[0],device='cuda:0')
        c = c.unsqueeze(1)
        b = torch.cat((b, c), 1)
        one = (a - b) == 1
        result = result + one

        # （0，1）
        b = a[:, :-1]
        c = torch.zeros(a.shape[0],device='cuda:0')
        c = c.unsqueeze(1)
        b = torch.cat((c, b), 1)
        one = (a - b) == 1
        result = result + one

        # [1][0]
        b = a[:-1, :]
        c = torch.zeros(a.shape[1],device='cuda:0')
        c = c.unsqueeze(0)
        b = torch.cat((c, b), 0)
        one = (a - b) == 1
        result = result + one

        b = a[1:, :]
        c = torch.zeros(a.shape[1],device='cuda:0')
        c = c.unsqueeze(0)
        b = torch.cat((b, c), 0)
        one = (a - b) == 1
        result = result + one
        result = torch.where(result > 1, obj_pix, result)

        return result

    def get_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        inputs = torch.argmax(inputs, dim=1).squeeze(0).squeeze(0)
        target = target.squeeze(0)
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        inputs = self.get_boundary(inputs)
        target = self.get_boundary(target)
        loss = self.get_loss(inputs,target)
        return loss
class FocalLoss(nn.Module):
    """
    gamma越大，置信度高的样子损失与置信度低的样本损失差距越大。
    """
    def __init__(self, alpha=0.5, gamma=2, weight=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.ce_fn = nn.CrossEntropyLoss(weight=self.weight)

    def forward(self, preds, labels):
        logpt = -self.ce_fn(preds, labels)
        pt = torch.exp(logpt)
        loss = -((1 - pt) ** self.gamma) * self.alpha * logpt
        # loss = torch.tensor(loss,requires_grad=True)
        return loss
def get_f1_score(gt_image, pre_image, num_class=2):
    mask = (gt_image >= 0) & (gt_image < num_class)
    label = num_class * gt_image[mask].astype('int') + pre_image[mask]
    count = np.bincount(label, minlength=num_class ** 2)
    confusion_matrix = count.reshape(num_class, num_class)
    smooth = 1e-10
    precision = confusion_matrix[1][1] / (confusion_matrix[1][1] + confusion_matrix[0][1] + smooth)
    recall = confusion_matrix[1][1] / (confusion_matrix[1][1] + confusion_matrix[1][0] + smooth)
    f1_score = 2 * precision * recall / (precision + recall + smooth)
    return f1_score

def calculate_metric_percase(pred, gt):
    #统计pred和gt中是否有该类的目标
    pred[pred > 0] = 1
    gt[gt > 0] = 1

    #gt有该类别，pre有该类别，则计算，pred,gt必须均不全为0
    if pred.sum() > 0 and gt.sum()>0:
        dice = metric.binary.dc(pred, gt)
        union = np.logical_or(gt, pred)
        intersection = np.logical_and(gt, pred)
        iou = np.sum(intersection) / np.sum(union)
        hd95 = metric.binary.hd95(pred, gt)
        f1_score = get_f1_score(gt,pred)
        if dice==0 and iou==0:
            return 0, 0, 0, 0
        return dice, iou, hd95,f1_score
    else:
        return 0, 0, 0,0
def getmAP(pre_image,gt_image):
    a = np.abs(pre_image-gt_image).astype('int')
    ap = (np.bincount(a.flatten())[0])/ pre_image.size
    return ap

def get_nAP(pre,gt):
    # 阴性像素准确度、
    if np.sum(gt==1)==0:
        #说明没有这个类别信息
        return 0
    else:
        gt [gt!=1]=3
        pre[pre!=1]=5
        temp= pre-gt
        nAP = np.sum(temp==0)/np.sum(gt==1)
        return nAP

def get_pAP(pre,gt):
    # 阳性像素准确度
    if np.sum(gt==2)==0:
        return 0
    else:
        gt[gt != 2] = 4
        pre[pre != 2] = 5
        temp = pre - gt
        pAP = np.sum(temp == 0) / np.sum(gt == 2)
        return pAP
def get_tps(pre):
    p = np.sum(pre==2)
    n = np.sum(pre==1)
    if p==0:
        #无阳性
        return p,n,0
    else:
        tps = p/(p+n)
        return p,n,tps

def get_wp(gt,pre):
    #只要相邻，就正确
    temp = np.abs(gt-pre)
    temp = temp.astype(np.float)
    temp[temp>1]=-1
    temp[temp!=-1]=1
    temp[temp==-1]=0
    wp = np.mean(temp)
    return wp

def get_NP(gt,pre):
    #判断目标和背景
    pre[pre==2]=1
    gt[gt==2]=1
    temp = np.abs(gt-pre)
    wp = (temp.size-np.sum(temp))/temp.size
    return wp

def pd_toexcel(data,filename): # pandas库储存数据到excel
    # a=[i + 1 for i in range(len(data[0]))]

    p= [np.format_float_positional(float(format(i/(959*462),'.2g'))) for i in data[1]]
    n= [np.format_float_positional(float(format(i/(959*462),'.2g')))  for i in data[2]]
    tps = [np.format_float_positional(float(format(i,'.2g'))) for i in data[3]]

    dfData = { # 用字典设置aDataFrame所需数据
        '图片编号':data[0],
        'PD-L1阳性表达占比': p,
        'PD-L1阳性表达像素个数': data[1],
        'PD-L1阴性表达占比':n,
        'PD-L1阴性表达像素个数':data[2],
        'TPS分数':tps
    }
    df = pd.DataFrame(dfData) # 创建DataFrame
    df.to_excel(filename,index=False) # 存表，去除原始索引列（0,1,2...）

def test_single_volume(image, label, net, classes, patch_size=[256, 256], test_save_path=None, case=None, z_spacing=1):
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
    _,x, y = image.shape
    if x != patch_size[0] or y != patch_size[1]:
        #缩放图像符合网络输入
        image = zoom(image, (1,patch_size[0] / x, patch_size[1] / y), order=3)
    input = torch.from_numpy(image).unsqueeze(0).float().cuda()
    net.eval()
    with torch.no_grad():
        # a = net(input).cpu().detach().numpy()
        temp = torch.softmax(net(input), dim=1)
        # zero = torch.zeros_like(temp)
        #置信度小于0.95时，置为0
        # temp = torch.where(temp<0.95,zero,temp)

        #处理输出
        out = torch.argmax(temp, dim=1).squeeze(0)
        out = out.cpu().detach().numpy()
        if x != patch_size[0] or y != patch_size[1]:
            #缩放图像至原始大小
            prediction = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
        else:
            prediction = out

    metric_list = []

    #计算AP(未修改pre和label)
    AP = getmAP(prediction,label)
    #只要相邻就正确（未改变值）
    WP = get_wp(label,prediction)
    #阴性像素准确率
    n_AP=get_nAP(prediction.copy(),label.copy())
    #阳性像素准确率
    p_AP=get_pAP(prediction.copy(),label.copy())
    #TPS
    tps_p,tps_n,tps = get_tps(prediction.copy())
    #只判断目标和背景是否预测正确，copy深拷贝，不然会改变数组的值
    NP = get_NP(label.copy(),prediction.copy())
    temp = np.unique(prediction)
    #prediction的值为类别数
    for i in range(0, classes):
        temp = calculate_metric_percase(prediction == i, label == i)
        metric_list.append(temp)
    if (0,0,0,0) in metric_list:
        metric_list.remove((0, 0, 0, 0))
    metric_list = [np.mean(metric_list, axis=0)]

    metric_list[0] = np.append(metric_list[0],[AP,WP,NP,n_AP,p_AP,tps_p,tps_n,tps])

    if test_save_path is not None:
        #直接保存灰度图
        prediction[prediction==1]=125
        prediction[prediction==2]=255
        prediction = Image.fromarray(np.uint8(prediction))
        prediction.save(test_save_path+'/'+case+'.png')

    return metric_list

def reset_log(log_path):
    import logging
    fileh = logging.FileHandler(log_path, 'a')
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fileh.setFormatter(formatter)
    log = logging.getLogger()  # root logger
    for hdlr in log.handlers[:]:  # remove all old handlers
        log.removeHandler(hdlr)
    log.addHandler(fileh)
    log.setLevel(logging.DEBUG)