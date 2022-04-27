'''
将以文件夹形式存储的数据集，以文本列表的形式读取出来
例如，一个数据集放在“DS”文件夹中，文件夹里面有两个文件夹，“DS_org”表示原始图像，“DS_an”表示
低照度图像，那么数据集会形成两个列表，分别表示两种图像的文件名列表
'''
import os, glob
import numpy as np
from PIL import Image
import collections
import matplotlib.pyplot as plt


img_width = 256  # 目标图片的行数
img_height = 256  # 目标图片的列数
imgSizeNum = img_width * img_height  # 目标图片的总数据量
img_channel = 3  #图像的通道数

def MakeDataSet(path, dataname, pair=['dim', 'rgb']):
    '''
    :param path:表示数据集的文件夹名字
    :param dataname:表示数据集的名字
    :param pair:数据集的队标
    :return:返回两个列表，每个元素是成对的数据集中的输入X和标签label
    '''
    dim_path = path + '/' + dataname + '/' + pair[0]
    rgb_path = path + '/' + dataname + '/' + pair[1]
    dim_list = os.listdir(dim_path)
    rgb_list = os.listdir(rgb_path)
    return dim_list, rgb_list

def LoadImg(path, filename):
    fullpath = path + '/' + filename
    img = Image.open(fullpath)
    img = (np.array(img, dtype=np.float32)) / 255
    img = np.array(img[np.newaxis, :, :, :])
    return img

class ListDataSet:
    '''
       pair表示的是数据集原始图像和光照不足图像的文件夹后缀。
       例如，原始图像放在“dataset_dim”，光照不足图像放在“dataset_rgb”，则pair=['dim','rgb']
       '''
    def __init__(self,path='../Dataset/dim2rgb_img',
                 dataname='dimrgbpair256',pair=['dim','rgb']):
        self.list=MakeDataSet(path,dataname,pair)
        self.start_index = 0
        self.train_num=len(self.list[0])
        self.path=path
        self.dataname=dataname
        self.pair=pair
        img0_path = self.path+'/'+self.dataname+'/'+self.pair[0]
        img0 = Image.open(img0_path + '/' + self.list[0][0])
        img_size = img0.size
        self.img_w = img_size[1]
        self.img_h = img_size[0]


    def GetNextBatch(self,batchSize):
        '''
        读取batchSize的训练数据，返回训练数据的X和Label
        :param batchSize:读取数量
        :return:两个shape=[batchSize,ImgH,ImgW,ImgC]的numpy数组，一个表示X，一个表示Label
        '''
        # Define the returned data batches
        Data = collections.namedtuple('Data', ' dimImgs, dimNames, rgbImgs, rgbNames')

        if(self.start_index+batchSize>self.train_num):
            endIndex=self.train_num
        else:
            endIndex=self.start_index+batchSize
        input_path=self.path+'/'+self.dataname+'/'+self.pair[0]
        label_path=self.path+'/'+self.dataname+'/'+self.pair[1]
        temp_input=np.zeros(shape=[batchSize,self.img_w,self.img_h,img_channel])
        temp_label=np.zeros(shape=[batchSize,self.img_w,self.img_h,img_channel])
        temp_input_name = []
        temp_label_name = []
        for i in range(self.start_index,endIndex):
            input=LoadImg(input_path,self.list[0][i])
            label=LoadImg(label_path,self.list[1][i])
            temp_input[i-self.start_index,:,:,:]=input
            temp_label[i-self.start_index,:,:,:]=label
            temp_input_name.append(self.list[0][i])
            temp_label_name.append(self.list[1][i])
        self.start_index=(endIndex)%self.train_num
        return Data(
            dimImgs = temp_input,
            dimNames= temp_input_name,
            rgbImgs = temp_label,
            rgbNames=temp_label_name
        )

class TestDataSet:
    def __init__(self,path, img_c =3):
        self.start_index = 0
        self.list = os.listdir(path)
        self.test_num=len(self.list)
        self.path=path
        self.img_c=img_c
        img0 = Image.open(path + '/' + self.list[0])
        img_size = img0.size
        self.img_w = img_size[1]
        self.img_h = img_size[0]

    def GetNextBatch(self,batchSize):
        '''
        读取batchSize的训练数据，返回数据X
        :param batchSize:读取数量
        :return:一个shape=[batchSize,ImgH,ImgW,ImgC]的numpy数组，表示输入数据X
        '''
        # Define the returned data batches
        Data = collections.namedtuple('Data', ' xImgs, xName')

        if(self.start_index+batchSize>self.test_num):
            endIndex=self.test_num
        else:
            endIndex=self.start_index+batchSize

        temp_name = []
        for i in range(self.start_index,endIndex):
            fullpath = self.path + '/' + self.list[i]
            img = Image.open(fullpath)
            img_size = img.size
            temp_input = np.zeros(shape=[batchSize, img_size[1], img_size[0], self.img_c])
            # print('img.size = ', img.size)
            # print('temp_input.shape = ', temp_input.shape)

            img = (np.array(img, dtype=np.float32))
            # img = (np.array(img, dtype=np.float32)) / 255
            img = np.array(img[np.newaxis, :, :, :])
            temp_input[i-self.start_index,:,:,:]=img
            temp_name.append(self.list[i])

        self.start_index=(endIndex)%self.test_num
        return Data(
            xImgs = temp_input,
            xName = temp_name
        )


class LoadDataSet:
    '''
       pair表示的是数据集原始图像和光照不足图像的文件夹后缀。
       例如，原始图像放在“dataset_dim”，光照不足图像放在“dataset_rgb”，则pair=['dim','rgb']
       '''

    def __init__(self, path='../Dataset/dim2rgb_img',
                 dataname='dimrgbpair256', pair=['dim', 'rgb'],
                 img_c=3):
        self.list = MakeDataSet(path, dataname, pair)
        self.start_index = 0
        self.data_num = len(self.list[0])
        self.path = path
        self.dataname = dataname
        self.pair = pair
        self.path = path
        self.img_c = img_c
        img0 = Image.open(path + '/' + dataname + '/'+ self.pair[0] + '/'+ self.list[0][0])
        img_size = img0.size
        self.img_w = img_size[0]
        self.img_h = img_size[1]

    def GetNextBatch(self, batchSize):
        '''
        读取batchSize的训练数据，返回训练数据的X和Label
        :param batchSize:读取数量
        :return:两个shape=[batchSize,ImgH,ImgW,ImgC]的numpy数组，一个表示X，一个表示Label
        '''
        # Define the returned data batches
        Data = collections.namedtuple('Data', 'inputImgs, inputNames, tarImgs, tarNames')

        if (self.start_index + batchSize > self.data_num):
            endIndex = self.data_num
        else:
            endIndex = self.start_index + batchSize

        input_path = self.path + '/' + self.dataname + '/' + self.pair[0]
        label_path = self.path + '/' + self.dataname + '/' + self.pair[1]
        temp_input_name = []
        temp_label_name = []
        for i in range(self.start_index, endIndex):
            fullpath = input_path + '/' + self.list[0][i]
            img = Image.open(fullpath)
            img_size = img.size
            temp_input = np.zeros(shape=[batchSize, img_size[1], img_size[0], self.img_c])
            temp_label = np.zeros(shape=[batchSize, img_size[1], img_size[0], self.img_c])
            input = LoadImg(input_path, self.list[0][i])
            label = LoadImg(label_path, self.list[1][i])
            temp_input[i - self.start_index, :, :, :] = input
            temp_label[i - self.start_index, :, :, :] = label
            temp_input_name.append(self.list[0][i])
            temp_label_name.append(self.list[1][i])
        self.start_index = (endIndex) % self.data_num
        return Data(
            inputImgs=temp_input,
            inputNames=temp_input_name,
            tarImgs=temp_label,
            tarNames=temp_label_name
        )

# _gan_inData_path = 'D:/766QLL/dim2rgb_imgs'
# load_pairImgs = ListDataSet(path=_gan_inData_path, dataname="dim2rgb_train_256" )
# load_pairValis = ListDataSet(path=_gan_inData_path, dataname="dim2rgb_valid_256" )

# print("load_pairImgs", load_pairImgs.list)

# dim_list, rgb_list = MakeDataSet(path=_gan_inData_path,
#                                  dataname="dim2rgb_train_256",
#                                  pair=['dimImgs', 'rgbImgs'])

# print(dim_list)



