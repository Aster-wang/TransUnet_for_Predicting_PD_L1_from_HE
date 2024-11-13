import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
import torch
import numpy as np
from PIL import Image
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from pytorch_grad_cam import GradCAM
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
import argparse
from scipy.ndimage.interpolation import zoom
import cv2
class SegmentationModelOutputWrapper(torch.nn.Module):
    def __init__(self, model):
        super(SegmentationModelOutputWrapper, self).__init__()
        self.model = model
    def forward(self, x):
        return self.model(x)["out"]
class SemanticSegmentationTarget:
    def __init__(self, category, mask):
        self.category = category
        self.mask = torch.from_numpy(mask)
        if torch.cuda.is_available():
            self.mask = self.mask.cuda()

    def __call__(self, model_output):
        model_output = model_output.squeeze(0)
        return (model_output[self.category, :, :] * self.mask).sum()

parser = argparse.ArgumentParser()
parser.add_argument('--img_size', type=int,
                    default=224, help='input patch size of network input')#transunet=400，DAF=224
parser.add_argument('--num_classes', type=int,
                    default=3, help='output channel of network')
parser.add_argument('--n_skip', type=int,
                    default=3, help='using number of skip-connect, default is num')
parser.add_argument('--vit_name', type=str,
                    default='R50-ViT-B_16', help='select one vit model')
parser.add_argument('--vit_patches_size', type=int,
                    default=16, help='vit_patches_size, default is 16')
args = parser.parse_args()

image_url = "../../data/npz/1834236-2-PD-L1--10-HE-6-2.npz"
'''1833261-1-PD-L1--10-HE-13-4
1833261-1-PD-L1--10-HE-41-2
1834085-PD-L1--10-HE-19-2
1834236-2-PD-L1--10-HE-21-2 p
2138440-1-PD-L1--10-HE-4-4
1833261-1-PD-L1--10-HE-48-3
1834085-PD-L1--10-HE-72-3
1834085-PD-L1--10-HE-64-1
1834236-2-PD-L1--10-HE-6-2'''
data = np.load(image_url)
image, label = data['image'], data['label']
# rgb_img = np.float32(image) / 255   #??为什么255，结果还不变
rgb_img = np.float32(image)
img_ori = rgb_img
'''input_tensor_orisize = torch.Tensor(rgb_img).unsqueeze(0).permute(0,3,1,2).cuda()'''

x, y, _ = rgb_img.shape
if x != args.img_size or y != args.img_size:
    rgb_img = zoom(rgb_img, (args.img_size / x, args.img_size / y, 1), order=3)
temp_img = torch.tensor(rgb_img).unsqueeze(0).permute(0,3,1,2)

config_vit = CONFIGS_ViT_seg[args.vit_name]
config_vit.n_classes = args.num_classes
config_vit.n_skip = args.n_skip
if args.vit_name.find('R50') != -1:  # 如果包含，则返回第一次出现该字符串的索引；反之，则返回 -1。
    config_vit.patches.grid = (int(args.img_size / args.vit_patches_size), int(args.img_size / args.vit_patches_size))
model = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes)
model.load_state_dict(torch.load('./model/size224_epo200_bs16_lr0.0007_transunet-769/epoch_91_0.85.pth'))
model = model.eval()

if torch.cuda.is_available():
    model = model.cuda()
    input_tensor = temp_img.cuda()
output= model(input_tensor)

output = output.cpu().detach().numpy()
'''output = zoom(output, (1,1,x / args.img_size,y / args.img_size), order=0)'''
output = torch.tensor(output)
normalized_masks = torch.nn.functional.softmax(output, dim=1)
fore_mask = normalized_masks[0, :, :, :].argmax(axis=0).detach().cpu().numpy()
'''看看输出对不对'''
# fore_mask[fore_mask == 1] = 125
# fore_mask[fore_mask == 2] = 255
# img = Image.fromarray(np.uint8(fore_mask))
# img.show()
sem_classes = ['__background__', 'negative', 'positive']
sem_class_to_idx = {cls: idx for (idx, cls) in enumerate(sem_classes)}
negative_category = sem_class_to_idx["negative"]
positive_category = sem_class_to_idx["positive"]

fore_mask_float = np.float32(fore_mask == positive_category)
'''验证类别结果'''
# fore_mask_uint8 = 255 * np.uint8(fore_mask)
# both_images = np.hstack((rgb_img.astype(np.uint8), np.repeat(fore_mask_uint8[:, :, None], 3, axis=-1)))
# img = Image.fromarray(both_images)
# img.show()

target_layers = [model.transformer.embeddings.hybrid_model.body.block1.unit3.conv3]#transformer.embeddings.CRA segmentation_head
# transformer.embeddings.hybrid_model.body.block3.unit9.conv3   block2.unit4.conv3   block1.unit3.conv3
targets = [SemanticSegmentationTarget(positive_category, fore_mask_float)]
with GradCAM(model=model,
             target_layers=target_layers,
             use_cuda=torch.cuda.is_available()) as cam:
    grayscale_cam = cam(input_tensor=input_tensor,
                        targets=targets)[0, :]
    grayscale_cam1 = grayscale_cam.copy()
    grayscale_cam1 = cv2.resize(grayscale_cam1, (y, x), cv2.INTER_AREA)
    cam_image = show_cam_on_image(img_ori/255, grayscale_cam1, use_rgb=True)

cam_image = Image.fromarray(cam_image)
cam_image.show()
cam_image.save('./1834236-2-PD-L1--10-HE-6-2_x2.png')