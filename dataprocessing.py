import os
import h5py
import numpy as np
from tqdm import tqdm
from skimage.io import imread
"""
从给定的文件夹中读取红外（IR）和可见光（VIS）图像，将它们转换为浮点数格式并进行归一化，然后将其分割成小的图像块（patches）。这些图像块随后被存储在HDF5（.h5）文件中，同时排除那些对比度较低的图像块。
"""

def get_img_file(file_name):
    imagelist = []
    for parent, dirnames, filenames in os.walk(file_name):
        for filename in filenames:
            if filename.lower().endswith(
                    ('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff', '.npy')):
                imagelist.append(os.path.join(parent, filename))
    return imagelist

# 将RGB图像转换为灰度图像   
def rgb2y(img):
    y = img[0:1, :, :] * 0.299000 + img[1:2, :, :] * 0.587000 + img[2:3, :, :] * 0.114000
    return y

# 将图像分割为图像块
def Im2Patch(img, win, stride=1):
    k = 0
    endc = img.shape[0]
    endw = img.shape[1]
    endh = img.shape[2]
    patch = img[:, 0:endw - win + 0 + 1:stride, 0:endh - win + 0 + 1:stride]
    TotalPatNum = patch.shape[1] * patch.shape[2]
    Y = np.zeros([endc, win * win, TotalPatNum], np.float32)
    for i in range(win):
        for j in range(win):
            patch = img[:, i:endw - win + i + 1:stride, j:endh - win + j + 1:stride]
            Y[:, k, :] = np.array(patch[:]).reshape(endc, TotalPatNum)
            k = k + 1
    return Y.reshape([endc, win, win, TotalPatNum])

# 通过计算图像亮度的某个百分位范围内的亮度差异来确定对比度
def is_low_contrast(image, fraction_threshold=0.1, lower_percentile=10,
                    upper_percentile=90):
    """Determine if an image is low contrast."""
    limits = np.percentile(image, [lower_percentile, upper_percentile])
    ratio = (limits[1] - limits[0]) / limits[1]
    return ratio < fraction_threshold

data_name = "MSRS_train"
img_size = 128  # patch size
stride = 200  # patch stride

IR_files = sorted(get_img_file(r"MSRS_train/train/ir"))
VIS_files = sorted(get_img_file(r"MSRS_train/train/vi"))

assert len(IR_files) == len(VIS_files)

# data_dir = '/MMIF-CDDFuse-main/data'
h5f = h5py.File(os.path.join('data',
                             f'{data_name}_imgsize_{img_size}_stride_{stride}.h5'),
                'w')
h5_ir = h5f.create_group('ir_patchs')
h5_vis = h5f.create_group('vis_patchs')
train_num = 0
# 遍历所有IR和VIS图像
for i in tqdm(range(len(IR_files))):
   # 读取VIS图像，转换为浮点数并归一化到[0,1]范围，并改变其维度顺序到[通道, 高度, 宽度] 
    I_VIS = imread(VIS_files[i]).astype(np.float32).transpose(2, 0, 1) / 255.  # [3, H, W] Uint8->float32
        # 将VIS图像从RGB转换为灰度
    I_VIS = rgb2y(I_VIS)  # [1, H, W] Float32
    I_IR = imread(IR_files[i]).astype(np.float32)[None, :, :] / 255.  # [1, H, W] Float32

    # 红外和可见光谱图像进行分块
    I_IR_Patch_Group = Im2Patch(I_IR, img_size, stride)
    I_VIS_Patch_Group = Im2Patch(I_VIS, img_size, stride)  # (3, 256, 256, 12)

    for ii in range(I_IR_Patch_Group.shape[-1]):
        bad_IR = is_low_contrast(I_IR_Patch_Group[0, :, :, ii])
        bad_VIS = is_low_contrast(I_VIS_Patch_Group[0, :, :, ii])
        # #检查IR和VIS图像块的对比度是否过低 
        if not (bad_IR or bad_VIS):
            avl_IR = I_IR_Patch_Group[0, :, :, ii]  # available IR
            avl_VIS = I_VIS_Patch_Group[0, :, :, ii]
            avl_IR = avl_IR[None, ...]
            avl_VIS = avl_VIS[None, ...]
	# 在HDF5文件中为IR和VIS图像块创建数据集
            h5_ir.create_dataset(str(train_num), data=avl_IR,
                                 dtype=avl_IR.dtype, shape=avl_IR.shape)
            h5_vis.create_dataset(str(train_num), data=avl_VIS,
                                  dtype=avl_VIS.dtype, shape=avl_VIS.shape)
            train_num += 1

h5f.close()

with h5py.File(os.path.join('/MMIF-CDDFuse-main/data',
                            f'{data_name}_imgsize_{img_size}_stride_{stride}.h5'), "r") as f:
    for key in f.keys():
        print(f[key], key, f[key].name)
