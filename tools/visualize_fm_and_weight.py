from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
from tools import visualize_fm_utils
from config import config_mrvae_v2_2 as cfg_v2_2
from tools import ops
import cv2

# 代码源 https://blog.csdn.net/u010358677/article/details/70578572
#        https://raw.githubusercontent.com/grishasergei/conviz/master/conviz.py
def plot_conv_weights(weights, plot_dir, name, channels_all=True, filters_all=True, channels=[0], filters=[0]):
    """
    Plots convolutional filters
    :param weights: numpy array of rank 4
    :param name: string, name of convolutional layer
    :param channels_all: boolean, optional
    :return: nothing, plots are saved on the disk
    """

    w_min = np.min(weights)
    w_max = np.max(weights)

    # make a list of channels if all are plotted
    if channels_all:
        channels = range(weights.shape[2])

    # get number of convolutional filters
    if filters_all:
        num_filters = weights.shape[3]
        filters = range(weights.shape[3])
    else:
        num_filters = len(filters)

    # get number of grid rows and columns
    grid_r, grid_c = visualize_fm_utils.get_grid_dim(num_filters)

    # create figure and axes
    fig, axes = plt.subplots(min([grid_r, grid_c]),
                             max([grid_r, grid_c]))

    # iterate channels
    for channel_ID in channels:
        # iterate filters inside every channel
        if num_filters == 1:
            img = weights[:, :, channel_ID, filters[0]]
            axes.imshow(img, vmin=w_min, vmax=w_max, interpolation='nearest', cmap='seismic')
            # remove any labels from the axes
            axes.set_xticks([])
            axes.set_yticks([])
        else:
            for l, ax in enumerate(axes.flat):
                # get a single filter
                img = weights[:, :, channel_ID, filters[l]]
                # put it on the grid
                ax.imshow(img, vmin=w_min, vmax=w_max, interpolation='nearest', cmap='seismic')
                # remove any labels from the axes
                ax.set_xticks([])
                ax.set_yticks([])
        # save figure
        plt.savefig(os.path.join(plot_dir, '{}-{}.png'.format(name, channel_ID)), bbox_inches='tight')


def plot_conv_output(conv_img, plot_dir, name, filters_all=True, filters=[0]):
    w_min = np.min(conv_img)
    w_max = np.max(conv_img)

    # get number of convolutional filters
    if filters_all:
        num_filters = conv_img.shape[3]
        filters = range(conv_img.shape[3])
    else:
        num_filters = len(filters)

    # get number of grid rows and columns
    grid_r, grid_c = visualize_fm_utils.get_grid_dim(num_filters)

    # create figure and axes
    fig, axes = plt.subplots(min([grid_r, grid_c]),
                             max([grid_r, grid_c]))

    # iterate filters
    if num_filters == 1:
        img = conv_img[0, :, :, filters[0]]
        axes.imshow(img, vmin=w_min, vmax=w_max, interpolation='bicubic', cmap=cm.hot)
        # remove any labels from the axes
        axes.set_xticks([])
        axes.set_yticks([])
    else:
        for l, ax in enumerate(axes.flat):
            # get a single image
            img = conv_img[0, :, :, filters[l]]
            # img = img * 255
            img = img
            # put it on the grid
            ax.imshow(img, vmin=w_min, vmax=w_max, interpolation='bicubic', cmap=cm.hot)
            # remove any labels from the axes
            ax.set_xticks([])
            ax.set_yticks([])
    # save figure
    plt.savefig(os.path.join(plot_dir, '{}.png'.format(name)), bbox_inches='tight')
    return img

def visualize_fm_sum(conv_img, filters=[0]):
    '''
    将一层conv的特征图可视化：每个通道叠加后进行可视化
    :param conv_img:
    :param filters_all:
    :param filters:
    :return:
    '''
    num_filters = len(filters)

    # get number of grid rows and columns
    grid_r, grid_c = visualize_fm_utils.get_grid_dim(num_filters)

    # create figure and axes
    fig, axes = plt.subplots(min([grid_r, grid_c]),
                             max([grid_r, grid_c]))

    # iterate filters
    if num_filters == 1:
        img = conv_img[0, :, :, filters[0]]

    else:
        for l, ax in enumerate(axes.flat):
            # get a single image
            img = conv_img[0, :, :, filters[l]]

    return img

def get_fm_name(fm):
    str_list = fm.split('/')
    fm_name = ''
    for i in str_list[:-1]:
        fm_name = fm_name + '_' + i
    return fm_name[1:]

def visualize_fm_and_weight(feature_maps_list, fms, outpath, global_step):
    '''
    将一层卷积结果的每个通道的特征图都保存
    :param fm_names:
    :param fms:
    :param outpath:
    :param global_step:
    :return:
    '''
    print('=' * 50)
    print('''plot {}th step's feature maps'''.format(global_step))
    print('-' * 50)
    i = 0
    for fm in feature_maps_list:
        fm_name = get_fm_name(fm.name)
        print('''conv name  : {}\t\t\t\t\tchannel's num: {}'''.format(fm_name, fms[i].shape[3]))
        print('''conv shape : {}'''.format(fms[i].shape))
        # outpath2 = os.path.join(outpath, 'feature_map4', str(global_step), str(fm_name))
        outpath2 = os.path.join(outpath, 'feature_map4', str(global_step))
        fms_path = outpath2 + '\\' + fm_name + '.jpg'
        fm_sum_outpath = os.path.join(outpath2, 'fms_sum','{}.jpg'.format(fm_name))
        print("save path: {}".format(fms_path))
        ops.create_file(os.path.join(outpath2, 'fms_sum'))
        # ================================================
        # fm_channel_combination = []
        #
        # for j in range(fms[i].shape[3]):
        #     # fm_channel_img = plot_conv_output(conv_img=fms[i], plot_dir=outpath2, name=str(j), filters_all=False,
        #     #                                   filters=[j])
        #     fm_channel_img = visualize_feature_map(fms[i], outpath=outpath2,fm_name=fm_name)
        #     # fm_channel_img = visualize_fm_sum(conv_img=fms[i], filters=[j])
        #     fm_channel_combination.append(fm_channel_img)
        # fm_sum = sum(one for one in fm_channel_combination)  # 一层conv所有通道的叠加
        # cv2.imwrite(fullpath, fm_sum)
        # ================================================
        visualize_feature_map(fms[i], fms_path=fms_path,fm_sum_outpath=fm_sum_outpath, fm_name=fm_name)
        i += 1
        print('''successfully drawing {}'s feature map\n'''.format(fm_name))
        print("*" * 30)


# 代码源： https://blog.csdn.net/missyougoon/article/details/85645195
def visualize_feature_map(feature_batch, fms_path,fm_sum_outpath, fm_name):
    '''
    创建特征子图，创建叠加后的特征图
    :param feature_batch: 一个卷积层所有特征图
    :return:
    '''
    feature_map = np.squeeze(feature_batch, axis=0)

    feature_map_combination = []
    plt.figure(num=1,figsize=(8, 7))

    # 取出 featurn map 的数量，因为特征图数量很多，这里直接手动指定了。
    num_pic = feature_map.shape[2]

    # row, col = visualize_fm_utils.get_row_col(25)
    row, col = visualize_fm_utils.get_row_col(num_pic)
    # 将 每一层卷积的特征图，拼接层 5 × 5
    for i in range(0, num_pic):
        feature_map_split = feature_map[:, :, i]
        feature_map_combination.append(feature_map_split)
        plt.subplot(row, col, i+1)
        plt.imshow(feature_map_split)
        plt.axis('off')

    plt.savefig(fms_path) # 保存图像到本地
    plt.show()
    # plt.close()

    # 各特征图1：1叠加

    feature_map_sum = sum(ele for ele in feature_map_combination)
    # cv2.imwrite(os.path.join(outpath,'{}_sum2.png'.format(fm_name)),feature_map_sum)
    # feature_map_sum_shape = feature_map_sum.shape
    # plt.figure(num=2,figsize=feature_map_sum_shape)
    # fig = plt.gcf()
    # fig.set_size_inches(512,512)

    plt.imshow(feature_map_sum)
    plt.axis('off')

    plt.savefig(fm_sum_outpath)
    # plt.show()
    # fig.savefig(fm_sum_outpath)

def visualize_feature_map_sum(feature_batch):
    '''
    将每张子图进行相加
    :param feature_batch:
    :return:
    '''
    feature_map = np.squeeze(feature_batch, axis=0)

    feature_map_combination = []

    # 取出 featurn map 的数量
    num_pic = feature_map.shape[2]

    # 将 每一层卷积的特征图，拼接层 5 × 5
    for i in range(0, num_pic):
        feature_map_split = feature_map[:, :, i]
        feature_map_combination.append(feature_map_split)

    # 按照特征图 进行 叠加代码
    feature_map_sum = sum(one for one in feature_map_combination)

    plt.imshow(feature_map_sum)
    #plt.savefig('./mao_feature/feature_map_sum2.png') # 保存图像到本地
    plt.show()
