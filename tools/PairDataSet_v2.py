'''
将以文件夹形式存储的数据集，以文本列表的形式读取出来
例如，一个数据集放在“DS”文件夹中，文件夹里面有两个文件夹，“DS_org”表示原始图像，“DS_an”表示
低照度图像，那么数据集会形成两个列表，分别表示两种图像的文件名列表
'''

import os
import numpy as np
from PIL import Image
# import matplotlib.pyplot as plt
import tensorflow as tf
img_width=512
img_height=512
img_channel=3

def MakeDataSet(path='G:/cusdata',dataname='val2017256by256_train',pair=['y','x']):
	'''
	读取路径下的所有图片
	:param path:表示数据集的文件夹名字
	:param dataname:表示数据集的名字
	:param pair:数据集的队标
	:return:返回两个列表，每个元素是成对的数据集中的输入X和标签label
	'''
	# org_path=path+'/'+dataname+'_'+pair[0]
	# an_path=path+'/'+dataname+'_'+pair[1]

	org_path=path+'/'+dataname+'/' + pair[0]  # 正常照度路径
	an_path=path+'/'+dataname+'/' + pair[1]   # 低照度路径
	org_list=os.listdir(org_path)       # 获取所有的正常照度图片
	an_list=os.listdir(an_path)
	return org_list,an_list

def LoadImg(path='dim2rgb_train_256',filename='',img_width='512',img_height='512'):
	fullpath=path+'/'+filename
	img = Image.open(fullpath)
	img = img.resize((img_width,img_height))

	img=(np.array(img, dtype=np.float32)) / 255
	img=np.array(img[np.newaxis,:,:,:])
	return img

def LoadImgFromPath(path='dim2rgb_train_256',imgw=512,imgh=512,img_num=0):
	'''
	读取path下的img_num张图片,需要保证该目录下所有图片大小通道数一致，若img_num==0，则读取该路径下所有图片
	:param path:图片路径
	:param img_num:读取的图片数量
	:return:一个numpy数组，形如[n,w,h,c]
	'''
	img_list = os.listdir(path)
	# print(img_list)
	if img_num==0:
		img_num=len(img_list)
		print('''images' numbers:   {}'''.format(img_num))

	imgs=np.zeros(shape=[img_num,imgw,imgh,3])
	for i in range(img_num):
		# fullpath=path+img_list[i]
		fullpath=os.path.join(path,img_list[i])
		img = Image.open( fullpath )
		img = img.resize((imgw,imgh))
		img = (np.array( img, dtype=np.float32 )) / 255
		img = np.array( img[np.newaxis, :, :, :] )
		imgs[i,:,:,:]=img
	return imgs

def LoadImgFromPath3(path='dim2rgb_train_256',imgw=512,imgh=512,img_num=0):
	'''
	读取path下的img_num张图片,需要保证该目录下所有图片大小通道数一致，若img_num==0，则读取该路径下所有图片
	:param path:图片路径
	:param img_num:读取的图片数量
	:return:一个numpy数组，形如[n,w,h,c]
	'''
	img_list = os.listdir(path)
	img_size = []
	# print(img_list)
	if img_num==0:
		img_num=len(img_list)
		print('''images' numbers:   {}'''.format(img_num))

	imgs=np.zeros(shape=[img_num,imgw,imgh,3])
	for i in range(img_num):
		# fullpath=path+img_list[i]
		fullpath=os.path.join(path,img_list[i])
		img = Image.open( fullpath )
		img_size.append(img.size)
		img = img.resize((imgw, imgh))
		im_array = np.array(img)
		try:
			if len(im_array.shape) == 2:
				c = []
				for i in range(3):
					c.append(im_array)
					img = np.asarray(c)
					img = img.transpose([1, 2, 0])

			elif im_array.shape[2] == 4:
				img = Image.open(fullpath).convert("RGB")
				img = img.resize((imgw, imgh))
		except Exception as e:
			print('channels erro.')


		img = (np.array( img, dtype=np.float32 )) / 255
		img = np.array( img[np.newaxis, :, :, :] )
		imgs[i,:,:,:]=img
	return imgs, img_list, img_size

def LoadImgFromPath2(path='dim2rgb_train_256',imgw=512,imgh=512,img_num=0):
	'''
	读取path下的img_num张图片,需要保证该目录下所有图片大小通道数一致，若img_num==0，则读取该路径下所有图片
	:param path:图片路径
	:param img_num:读取的图片数量
	:return:一个numpy数组，形如[n,w,h,c]
	'''
	img_list = os.listdir(path)
	# print(img_list)
	if img_num==0:
		img_num=len(img_list)
		print('''images' numbers:   {}'''.format(img_num))

	imgs=np.zeros(shape=[img_num,imgw,imgh,3])
	for i in range(img_num):
		# fullpath=path+img_list[i]
		fullpath=os.path.join(path,img_list[i])
		img = Image.open( fullpath )
		img = img.resize((imgw,imgh))
		img = (np.array( img, dtype=np.float32 )) / 255
		img = np.array( img[np.newaxis, :, :, :] )
		imgs[i,:,:,:]=img
	return imgs, img_list

def LoadImgFromPath_N(path='dim2rgb_train_256',imgw=512,imgh=512,img_num=0,start_index=0):
	'''
	读取path下的img_num张图片，从start_index张开始读
	:param path:图片路径
	:param img_num:读取的图片数量
	:param start_index:开始读取的图片索引
	:return:一个numpy数组，形如[n,w,h,c]
	'''
	img_list = os.listdir(path)
	if img_num==0: img_num=len(img_list)

	imgs=np.zeros(shape=[img_num,imgw,imgh,3])
	for i in range(img_num):
		fullpath=path+img_list[i+start_index]
		img = Image.open( fullpath )
		img = (np.array( img, dtype=np.float32 )) / 255
		img = np.array( img[np.newaxis, :, :, :] )
		imgs[i,:,:,:]=img

	return imgs

class ListDataSet:
	'''
	pair表示的是数据集原始图像和光照不足图像的文件夹后缀。
	例如，原始图像放在“dataset_x”，光照不足图像放在“dataset_y”，则pair=['x','y']
	'''
	def __init__(self,path='G:/cusdata',dataname='val2017256by256_train',pair=['y','x'],img_height=512,img_width=512,img_channel=3):
		self.list=MakeDataSet(path,dataname,pair)   # 读取所有图片
		self.start_index=0
		self.train_num=len(self.list[0])    # 训练图片数量
		self.path=path              # 数据集主路径
		self.dataname=dataname      # 数据集名称
		self.pair=pair              # 正常照度、低照度文件夹名称

		img0_path = self.path + '/' + self.dataname + '/' + self.pair[0]
		img0 = Image.open(img0_path + '/' + self.list[0][0])
		img_size = img0.size
		# self.img_height = img_size[1]
		# self.img_width = img_size[0]
		self.img_height = img_height
		self.img_width = img_width
		self.img_channel = img_channel

	def GetNextBatch(self,batchSize):
		'''
		读取batchSize的训练数据，返回训练数据的X和Label
		:param batchSize:读取数量
		:return:两个shape=[batchSize,ImgH,ImgW,ImgC]的numpy数组，一个表示X，一个表示Label
		'''
		# 终止下标，如果start_index + batchSize > train_num，则endIndex为train_num
		# 否则，为start_index + batchSize
		if(self.start_index+batchSize>self.train_num):
			endIndex=self.train_num
		else:
			endIndex=self.start_index+batchSize

		input_path=self.path+'/'+self.dataname+'/'+self.pair[1]   # 低照度图像路径
		label_path=self.path+'/'+self.dataname+'/'+self.pair[0]   # 正常照度图像路径
		temp_input=np.zeros(shape=[batchSize,self.img_height,self.img_width,self.img_channel])
		temp_label=np.zeros(shape=[batchSize,self.img_height,self.img_width,self.img_channel])
		for i in range(self.start_index,endIndex):
			input=LoadImg(input_path,self.list[1][i],self.img_width,self.img_height)
			label=LoadImg(label_path,self.list[0][i],self.img_width,self.img_height)
			temp_input[i-self.start_index,:,:,:]=input
			temp_label[i-self.start_index,:,:,:]=label

		self.start_index=(endIndex)%self.train_num  # 更新start_index
		return temp_input,temp_label




# '''
# 将以文件夹形式存储的数据集，以文本列表的形式读取出来
# 例如，一个数据集放在“DS”文件夹中，文件夹里面有两个文件夹，“DS_org”表示原始图像，“DS_an”表示
# 低照度图像，那么数据集会形成两个列表，分别表示两种图像的文件名列表
# '''
#
# import os
# import numpy as np
# from PIL import Image
# import matplotlib.pyplot as plt
#
# img_width=512
# img_height=512
# img_channel=3
#
# def MakeDataSet(path='G:/cusdata',dataname='val2017256by256_train',pair=['x','y']):
#     '''
#     :param path:表示数据集的文件夹名字
#     :param dataname:表示数据集的名字
#     :param pair:数据集的队标
#     :return:返回两个列表，每个元素是成对的数据集中的输入X和标签label
#     '''
#     org_path=path+'/'+dataname+'_'+pair[0]
#     an_path=path+'/'+dataname+'_'+pair[1]
#     org_list=os.listdir(org_path)
#     an_list=os.listdir(an_path)
#     return org_list,an_list
#
# def LoadImg(path='G:/cusdata/val2017256by256_train_x',filename=''):
#     fullpath=path+'/'+filename
#     img = Image.open(fullpath)
#     img=(np.array(img, dtype=np.float32)) / 255
#     img=np.array(img[np.newaxis,:,:,:])
#
#     return img
#
# # list=MakeDataSet()
# # print(len(list[0]))
#
# # org,an=MakeDataSet()
# # img1=LoadImg(filename=org[0])
# # img2=LoadImg(filename=org[1])
# # img3=LoadImg(filename=org[2])
# # tmp=np.concatenate([img1,img2],axis=0)
# # tmp=np.concatenate([tmp,img3],axis=0)
# #
# # print(np.shape(tmp))
# #
# # plt.imshow(tmp[0,:,:,:])
# # plt.show()
#
# # for i in range(100):
# #     print(org[i],'---',an[i])
#
# class ListDataSet:
#     '''
#     pair表示的是数据集原始图像和光照不足图像的文件夹后缀。
#     例如，原始图像放在“dataset_y”，光照不足图像放在“dataset_x”，则pair=['y','x']
#     '''
#     def __init__(self,path='G:/cusdata',dataname='val2017256by256_train',pair=['y','x']):
#         '''
#         :param path: 数据集文件夹路径
#         :param dataname: 光照不足文件夹和正常光照文件夹的通名,如data_y,data_x，通名就是data
#         :param pair:正常光照文件夹和光照不足文件夹的后缀，如['y','x']
#         '''
#         self.list=MakeDataSet(path,dataname,pair)
#         self.start_index=0
#         self.train_num=len(self.list[0])
#         self.path=path
#         self.dataname=dataname
#         self.pair=pair
#
#     def GetNextBatch(self,batchSize,isAutoSize=True):
#         '''
#         读取batchSize的训练数据，返回训练数据的X和Label
#         :param batchSize:读取数量
#         :return:两个shape=[batchSize,ImgH,ImgW,ImgC]的numpy数组，一个表示X，一个表示Label
#         '''
#         if (self.start_index + batchSize > self.train_num):
#             endIndex = self.train_num
#         else:
#             endIndex = self.start_index + batchSize
#         input_path=self.path+'/'+self.dataname+'_'+self.pair[1]
#         label_path=self.path+'/'+self.dataname+'_'+self.pair[0]
#         temp_input=np.zeros(shape=[batchSize,img_height,img_width,img_channel])
#         temp_label=np.zeros(shape=[batchSize,img_height,img_width,img_channel])
#         for i in range(self.start_index,endIndex):
#             input=LoadImg(input_path,self.list[1][i])
#             label=LoadImg(label_path,self.list[0][i])
#             temp_input[i-self.start_index,:,:,:]=input
#             #print(np.shape(label))
#             temp_label[i-self.start_index,:,:,:]=label
#
#         self.start_index=endIndex%self.train_num
#         return temp_input,temp_label

# dataset=ListDataSet()
# x,y=dataset.GetNextBatch(5)
# print(np.shape(x))