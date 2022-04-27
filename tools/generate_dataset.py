
import os
import cv2
import random
import numpy as np
from PIL import Image
from tools import ops
import matplotlib.pyplot as plt

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# def _img_read(img_path, img_w, img_h, is_resize=True):
#     img = cv2.imread(img_path)
#     if is_resize:
#         img = cv2.resize(src=img,dsize=(img_w, img_h))
#     # 像素值归一化\
#     img_distance = np.max(img) - np.min(img)
#     img = (img-np.min(img)) / img_distance
#
#     return img, img_distance

def _img_read(img_path, img_w, img_h, is_resize=True):
    # img = Image.open(img_path)
    img = cv2.imread(img_path)
    if is_resize:
        img = cv2.resize(src=img, dsize=(img_w, img_h))
        # img = img.resize((img_w, img_h))
    img_distance = np.max(img) - np.min(img)
    # img = ((np.array(img, dtype=np.float32)) - np.min(img)) / img_distance
    img = (img - np.min(img)) / img_distance

    return img, img_distance

def _Gaussian_Noise(img, means, sigma):
    NoiseImg = img
    rows = NoiseImg.shape[0]
    cols = NoiseImg.shape[1]
    for i in range(rows):
        for j in range(cols):
            # NoiseImg[i, j] = NoiseImg[i, j] + random.gauss(means, sigma/255 )
            NoiseImg[i, j] = NoiseImg[i, j] + random.gauss(means, sigma )
            # for k in range(channels):
            #     if NoiseImg[i, j] < np.min(img):
            #         NoiseImg = np.min(img)
            #     elif NoiseImg[i, j] > np.max(img):
            #         NoiseImg = np.max(img)

    NoiseImg = np.clip(NoiseImg,np.min(img),np.max(img))
    return NoiseImg

def _Gaussian_Noise2(src, means, sigma, percentage):
    '''
    :param src: 输入图像
    :param means: 均值
    :param sigma: 方差
    :param percentage: 百分比
    :return:
    '''
    NoiseImg = src
    NoiseNum = int(percentage * src.shape[0] * src.shape[1])
    for i in range(NoiseNum):
        # 返还一个迭代器
        # 把一张图片的像素用行和列表示的话，randX代表随机生成的行，randY代表随机生成的列
        # random.randint生成随机整数
        # 高斯噪声图片边缘不处理，故-1
        randX = random.randint(0, src.shape[0] - 1)
        randY = random.randint(0, src.shape[1] - 1)
        # 此处在原有像素灰度值上加上随机数
        NoiseImg[randX, randY] = NoiseImg[randX, randY] + random.gauss(means, sigma)
        # 若灰度值小于0则强制为0，若灰度值大于255则强制为255
        # if NoiseImg[randX, randY] < 0:
        #     NoiseImg[randX, randY] = 0
        # elif NoiseImg[randX, randY] > 255:
        #     NoiseImg[randX, randY] = 255
    NoiseImg = np.clip(NoiseImg, 0, 1.0)
    return NoiseImg

def generate_dataset(index,gamma_ranges, in_path, out_path, img_w, img_h, is_resize=True):
    num = 0
    if index == 1 or index == 2:
        num = 7
    else:
        num = 3
    for img_name in os.listdir(in_path):
        for j in range(num):  # 每张图片，在每个照度级别生成5张图片，4个照度级别，共20张

            img_path = os.path.join(in_path, img_name)
            img, img_distance = _img_read(img_path, img_w=img_w, img_h=img_h, is_resize=is_resize)
            # img = img.resize((img_w,img_h))
            # 图像处理
            sigmaX = random.uniform(0, 1.1)
            gamma = random.uniform(gamma_ranges[0], gamma_ranges[1])
            means = 0
            sigma = 0
            percentage = random.uniform(0.001, 0.005)
            # percentage = 0.02
            if gamma <= 3:
                sigma = 0.5 * np.power(np.e, gamma)  # 指数关系
                # sigma = 10*gamma        # 线性关系
            elif gamma > 3:
                sigma = 10
            print('gamma:{}     sigma:{}'.format(gamma, sigma))
            new_img = cv2.GaussianBlur(src=img, ksize=(9, 9), sigmaX=sigmaX)  # 高斯模糊处理
            new_img = np.power((new_img), gamma)
            # new_img = _Gaussian_Noise(new_img, means=means, sigma=sigma)
            new_img = _Gaussian_Noise2(new_img, means=means, sigma=sigma, percentage=percentage)
            new_img = new_img * img_distance  # 像素值恢复到[0, 255]

            # 存储新图片
            # new_img_name = str(gamma) + '_' + str(sigma) + '_' + img_name[0:-4] + '_dim.jpg'
            new_img_name = img_name[0:-4] + str(round(gamma, 2)) + '_' + str(round(sigmaX, 2)) + '_' + str(
                round(sigma, 2)) + '_' + '_dim.jpg'
            # new_img_name = img_name[0:-4] + '_' + str(gamma)  + '_dim.jpg'
            new_out_path = os.path.join(out_path, new_img_name)
            cv2.imwrite(filename=new_out_path, img=new_img)
            print(img_name[0:-4] + ' has processed.')

def gamma_correction(gamma, in_path, out_path, img_w, img_h, is_resize=True):

    for img_name in os.listdir(in_path):
        img_path = os.path.join(in_path, img_name)
        img, img_distance = _img_read(img_path, img_w=img_w, img_h=img_h,is_resize=is_resize)
        # plt.ylim((0,250000))
        plt.hist(img.ravel(),256,[0,256])
        hist_name1 =  img_name[0:-4] + '_hist.jpg'
        hist_outpath1 = os.path.join(r'E:\WuXu\Dawn\MRVAE-Modification\data\hist\nor',hist_name1)
        plt.savefig(hist_outpath1)
        # 图像处理

        new_img = np.power((img), gamma)
        new_img = new_img * img_distance        # 像素值恢复到[0, 255]
        # plt.ylim((0, 250000))
        plt.hist(new_img.ravel(),256,[0,256])   # ravel，将多维数组降为一维数组

        # 存储新图片
        # new_img_name = str(gamma) + '_' + str(sigma) + '_' + img_name[0:-4] + '_dim.jpg'
        hist_name =  img_name[0:-4] + '_' + str(gamma) + '_hist.jpg'
        hist_outpath = os.path.join('E:\WuXu\Dawn\MRVAE-Modification\data\hist\dim',hist_name)
        plt.savefig(hist_outpath)

        new_img_name = img_name[0:-4] + '_' + str(gamma) + '_dim.jpg'
        new_out_path = os.path.join(out_path, new_img_name)
        cv2.imwrite(filename=new_out_path,img=new_img)
        print(img_name[0:-4] + ' has processed.')

def generate_dataset_batch():

    # in_path = r'E:\Detect_dataset\VOC2007\VOCtrainval_06-Nov-2007\VOCdevkit\VOC2007\JPEGImages'
    in_path = r'E:\WuXu\Dawn\MRVAE-Modification\tools\tp'
    out_path = r'E:\WuXu\Dawn\MRVAE-Modification\tools\tp2'
    # in_path = r'E:\WuXu\Dawn\MRVAE-Modification\data\img'
    # in_path = r'..\data\find_dataet\org'
    # in_path = r'E:\WuXu\Dawn\MRVAE-Modification\data\org_trainImgs_2000'
    # out_path = r'E:\Detect_dataset\VOC2007_ll\VOCtrainval_06-Nov-2007\VOCdevkit\VOC2007\JPEGImages'
    # out_path = r'..\data\Synthetic_train_set4'
    ops.create_file(out_path)

    img_w = 512
    img_h = 512

    # 相机曝光          伽马矫正系数
    # [-3,-2]           [3,4]
    # [-2,-1]           [2,3]
    # [-1,0]            [1,2]
    # [0,2]             [0.5,1]
    gamma = []
    # gamma.append([0.5,1])
    gamma.append([1.5,2])
    gamma.append([2,3])
    gamma.append([3,4])
    gamma.append([4,4.5])
    exposure_level = 4  # 曝光的等级数量，共四个
    for i in range(exposure_level):
        # full_out_path = os.path.join(out_path,str(i))
        # ops.create_file(full_out_path)
        generate_dataset(index=i,gamma_ranges=gamma[i], in_path=in_path, out_path=out_path,
                         img_w=img_w, img_h=img_h, is_resize=False)

def c():
    in_path = r'E:\WuXu\Dawn\MRVAE-Modification\data\img'
    out_path = r'E:\WuXu\Dawn\MRVAE-Modification\data\test2'
    ops.create_file(out_path)

    gamma = [0.3,0.5,0.8,1.0,1.2,1.5,1.7,2.0,2.2,2.5,2.7,3.0,3.2,3.5,3.7,4]

    img_w = 512
    img_h = 512
    for i in range(len(gamma)):
        gamma_correction(gamma[i],in_path,out_path,img_w,img_h,is_resize=True)

# c()
generate_dataset_batch()