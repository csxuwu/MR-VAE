import tensorflow as tf
import tensorflow.contrib.slim as slim
from tools import utils
from tools import ops
from choice import cfg_v2_1
from tools import development_kit as dk

'''
    原始的mrvae04 的结构，现在添加对每个卷积模块特征图的输出，只是添加了特征图的输出，结构上没有改
    
    最终使用的结构
'''

cfg = cfg_v2_1

def encoder_block(name, inputs, num_outputs, is_pool=True, scope=''):
    '''
    编码模块，返回池化之前的卷积输出，用于跳跃传递信息，和池化之后的输出，用于下一层编码
    :param name:
    :param inputs:
    :param num_outputs:
    :param is_pool:
    :return:
    '''
    with tf.variable_scope(name,reuse=tf.AUTO_REUSE):
        with slim.arg_scope([slim.conv2d],
                            kernel_size=3,
                            stride=1,
                            weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                            padding='SAME',
                            activation_fn=tf.nn.relu):
            with slim.arg_scope([slim.max_pool2d],
                                kernel_size=2,
                                stride=2,
                                padding='VALID',
                                ):
                '''卷积操作'''
                conv1 = slim.conv2d(inputs=inputs, num_outputs=num_outputs)
                conv2 = slim.conv2d(inputs=conv1, num_outputs=num_outputs)
                '''下采样操作'''
                if is_pool:
                    max_pool = slim.max_pool2d(conv2, scope=name)
                    print(name + ':{}'.format(conv2.get_shape()))
                    print(name + '_max_pool:{}'.format(max_pool.get_shape()))
                    return conv2, max_pool
                else:
                    print(name + ':{}'.format(conv2.get_shape()))
                    return conv2


def encoder(inputs, scope):
    '''
    参考U-Net编码结构
    :param inputs:
    :return:
    '''
    with tf.variable_scope('encoder'):
        print('========== Encoder ==========')
        # 返回池化前的卷积输出，作为跳跃链接到解码阶段；返回池化后的输出，作为下一层编码的输入
        encoder_conv1, max_pool1 = encoder_block(name='downsampling1', inputs=inputs, num_outputs=64)       ;tf.add_to_collection('feature_maps', encoder_conv1)
        encoder_conv2, max_pool2 = encoder_block(name='downsampling2', inputs=max_pool1, num_outputs=128)   ;tf.add_to_collection('feature_maps', encoder_conv2)
        encoder_conv3, max_pool3 = encoder_block(name='downsampling3', inputs=max_pool2, num_outputs=256)   ;tf.add_to_collection('feature_maps', encoder_conv3)
        encoder_conv4, max_pool4 = encoder_block(name='downsampling4', inputs=max_pool3, num_outputs=512)   ;tf.add_to_collection('feature_maps', encoder_conv4)
        # 最后一层，计算均值，不需要池化
        encoder_conv51 = encoder_block(name='z_mean', inputs=max_pool4, num_outputs=cfg.latent_dim, is_pool=False)     ;tf.add_to_collection('feature_maps', encoder_conv51)
        encoder_conv52 = encoder_block(name='z_variance', inputs=max_pool4, num_outputs=cfg.latent_dim, is_pool=False)  ;tf.add_to_collection('feature_maps', encoder_conv52)

        return encoder_conv1, encoder_conv2, encoder_conv3, encoder_conv4, encoder_conv51, encoder_conv52


def decoder_block(name, inputs, num_outputs, num_outputs_up=None, en_conv=None, is_transpose=False,
                  is_pixel_shuffler=False, is_resize=False, img_width=512, img_height=512, is_training=True):
    '''
    解码模块，参考U-Net
    :param name:
    :param inputs:
    :param num_outputs:卷积输出
    :param num_outputs_up:反卷积的输出
    :param en_conv:编码阶段传递过来的信息
    :param is_transpose:是否反卷积，默认使用反卷积上采样
    :param is_pixel_shuffler:是否使用子像素卷积进行上采样
    :param is_resize:是否使用resize进行上采样
    :param img_width:resize后的图像宽度
    :param img_height:resize后的图像高度
    :return:
    '''
    with tf.variable_scope(name):
        with slim.arg_scope([slim.conv2d_transpose],
                            kernel_size=2,
                            stride=2,
                            weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                            padding='SAME',
                            activation_fn=tf.nn.relu):
            with slim.arg_scope([slim.conv2d],
                                kernel_size=3,
                                stride=1,
                                weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                padding='SAME',
                                activation_fn=tf.nn.relu):
                if en_conv != None:
                    inputs = tf.concat(axis=3, values=[inputs, en_conv])
                # 卷积操作
                conv = slim.conv2d(inputs=inputs, num_outputs=num_outputs, scope=name + '_conv1')
                conv = slim.conv2d(inputs=conv, num_outputs=num_outputs, scope=name + '_conv2')
                # 上采样操作
                if is_transpose:  # 采用反卷积
                    t_conv1 = slim.conv2d_transpose(inputs=conv, num_outputs=num_outputs_up, scope=name + '_transpose')
                    print(name + ':{}'.format(conv.get_shape()))
                    print(name + '_transposel:{}'.format(t_conv1.get_shape()))
                    return t_conv1
                elif is_resize:  # 采用resize+conv
                    r_conv = tf.image.resize_images(images=conv, size=[img_width, img_height])
                    up_conv = slim.conv2d(inputs=r_conv, num_outputs=num_outputs_up, scope=name + '_resize')
                    print(name + ':{}'.format(conv.get_shape()))
                    print(name + '_resize:{}'.format(up_conv.get_shape()))
                    return up_conv
                elif is_pixel_shuffler:  # 采用子像素卷积
                    up_conv = utils.pixel_shuffler(conv, 2)
                    print(name + ':{}'.format(conv.get_shape()))
                    print(name + '_pixel_shuuffler:{}'.format(up_conv.get_shape()))
                    return up_conv
                else:  # 不进行上采样
                    print(name + ':{}'.format(conv.get_shape()))
                    return conv


def GR(inputs, conv1, conv2, conv3, conv4, up_w, up_h):
    '''
    构建全局特征
    :param inputs:
    :param conv1:
    :param conv2:
    :param conv3:
    :param conv4:
    :return:
    '''
    with tf.variable_scope('decoder_first'):
        with slim.arg_scope([slim.conv2d],
                            kernel_size=3,
                            stride=1,
                            weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                            padding='SAME',
                            activation_fn=tf.nn.relu):
            print('========== GR ==========')
            '''上采样'''
            up_w *= 2;up_h *= 2
            GR_upsampling1 = decoder_block(name='upsampling1', inputs=inputs, num_outputs=cfg.latent_dim,
                                           num_outputs_up=512, is_resize=True, img_width=up_w, img_height=up_h)
            tf.add_to_collection('feature_maps', GR_upsampling1)

            up_w *= 2;up_h *= 2
            GR_upsampling2 = decoder_block(name='upsampling2', inputs=GR_upsampling1, num_outputs=512,
                                           num_outputs_up=256, en_conv=conv4, is_resize=True, img_width=up_w,img_height=up_h)
            tf.add_to_collection('feature_maps', GR_upsampling2)

            up_w *= 2;up_h *= 2
            GR_upsampling3 = decoder_block(name='upsampling3', inputs=GR_upsampling2, num_outputs=256,
                                           num_outputs_up=128, en_conv=conv3, is_resize=True, img_width=up_w,img_height=up_h)
            tf.add_to_collection('feature_maps', GR_upsampling3)

            up_w *= 2;up_h *= 2
            GR_upsampling4 = decoder_block(name='upsampling4', inputs=GR_upsampling3, num_outputs=128,
                                           num_outputs_up=64, en_conv=conv2, is_resize=True, img_width=up_w,img_height=up_h)
            tf.add_to_collection('feature_maps', GR_upsampling4)

            '''输出'''
            de_conv = decoder_block(name='de_conv', inputs=GR_upsampling4, num_outputs=64, en_conv=conv1)
            de_out = slim.conv2d(inputs=de_conv, num_outputs=3, kernel_size=1, activation_fn=tf.nn.sigmoid)
            print('de_out:{}'.format(de_out.get_shape()))
            return de_out


def residual_block(no, inputs, inputs64=None):
    '''
    构造残差模块
    :param no:
    :param inputs:
    :param inputs64:第一次残差与输入相加时需要调整输入的通道数
    :return:
    '''
    name = 'residual_block' + str(no)
    with tf.variable_scope(name):
        with slim.arg_scope([slim.conv2d],
                            kernel_size=3,
                            stride=1,
                            weights_initializer=slim.xavier_initializer(),
                            padding='SAME',
                            activation_fn=None):  # BN后再使用激活函数
            '''第一个卷积模块：卷积 + BN + PReLU'''
            residual1_conv1 = slim.conv2d(inputs=inputs, num_outputs=64)
            residual1_conv1_BN = slim.batch_norm(residual1_conv1)
            residual1_conv1_PRelu = ops.prelu_tf(residual1_conv1_BN)

            '''第二个卷积模块：卷积 + BN + 跳跃  无激活函数'''
            residual1_conv2 = slim.conv2d(inputs=residual1_conv1_PRelu, num_outputs=64)
            residual1_conv2_BN = slim.batch_norm(residual1_conv2)
            if inputs64 != None:
                residual1_out = tf.add(inputs64, residual1_conv2_BN)  # 元素级别的相加，将输入与残差模块的输出相加
            else:
                residual1_out = tf.add(inputs, residual1_conv2_BN)  # 元素级别的相加，将输入与残差模块的输出相加
            print('{}st residuak:{}'.format(no, residual1_out.get_shape()))
            return residual1_out


def DR(inputs, de_conv1, is_training=True):
    '''
    第二阶段，参考SRResnet
    :param inputs:
    :param de_conv1:将编码的第一层输出传递过来
    :return:
    '''
    with tf.variable_scope('decoder_second'):
        with slim.arg_scope([slim.conv2d],
                            kernel_size=3,
                            stride=1,
                            weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                            padding='SAME',
                            activation_fn=None):
            print("========== DR ==========")
            '''5个跳跃块，跳跃时使用元素级别的相加'''
            inputs64 = slim.conv2d(inputs=inputs, num_outputs=64, kernel_size=1, stride=1,
                                   scope='inputs64')  # 输入数据是3通道，用1x1卷积变换通道数到,便于跳跃
            inputs_concat = tf.concat(axis=3, values=[inputs, de_conv1])
            inputs_concat_channels = inputs_concat.get_shape()[3]
            DR_residual1 = residual_block(1, inputs_concat, inputs64)   ;tf.add_to_collection('feature_maps', DR_residual1)
            DR_residual2 = residual_block(2, DR_residual1)              ;tf.add_to_collection('feature_maps', DR_residual2)
            DR_residual3 = residual_block(3, DR_residual2)              ;tf.add_to_collection('feature_maps', DR_residual3)
            DR_residual4 = residual_block(4, DR_residual3)              ;tf.add_to_collection('feature_maps', DR_residual4)
            DR_residual5 = residual_block(5, DR_residual4)              ;tf.add_to_collection('feature_maps', DR_residual5)

            '''普通卷积，输出'''
            '''普通卷积，输出'''
            DR_out64 = slim.conv2d(DR_residual5, num_outputs=64, scope='DR_out64')
            DR_out64_prelu = utils.prelu_tf(DR_out64, name='out64_prelu')       ;tf.add_to_collection('feature_maps', DR_out64_prelu)
            DR_out67 = slim.conv2d(DR_out64_prelu, num_outputs=inputs_concat_channels, kernel_size=1,scope='DR_out67')  # 调整通道数
            DR_out67_add = tf.add(inputs_concat, DR_out67)  # 将跳跃的输入通过跳跃传递过来

            DR_out32 = slim.conv2d(DR_out67_add, num_outputs=32, scope='DR_out32')
            DR_out32_prelu = utils.prelu_tf(DR_out32, name='DR_out32_prelu')    ;tf.add_to_collection('feature_maps', DR_out32_prelu)
            DR_out3 = slim.conv2d(DR_out32_prelu, num_outputs=3, kernel_size=1, scope='DR_out3')
            DR_out3_sigmoid = tf.nn.sigmoid(DR_out3, name='DR_out3_sigmoid')
            print('decoder_second_out:{}'.format(DR_out3_sigmoid.get_shape()))

            return DR_out3_sigmoid


def mrvae():
    t_image_ph = tf.placeholder(name='train_images', dtype=tf.float32,
                                shape=[cfg.batch_size, cfg.image_size, cfg.image_size, 3])
    t_org_image_ph = tf.placeholder(name='train_images', dtype=tf.float32,
                                    shape=[cfg.batch_size, cfg.image_size, cfg.image_size, 3])

    # 搭建mrvae
    with tf.name_scope(cfg.name):
        conv1, conv2, conv3, conv4, z_mean, z_variance = encoder(t_image_ph, scope='an')
        conv11, conv21, conv31, conv41, z_mean1, z_variance1 = encoder(t_org_image_ph, scope='org')
        z = utils.sampling(z_mean, z_variance)
        z1 = utils.sampling(z_mean1, z_variance1)
        z_w = z1.get_shape()[1]
        z_h = z1.get_shape()[2]
        decoder_first_out = GR(z, conv1, conv2, conv3, conv4, z_w, z_h)
        decoder_second_out = DR(decoder_first_out, conv1)

    with tf.name_scope('Loss'):
        L2 = tf.reduce_mean(tf.reduce_sum(tf.square(decoder_first_out - t_org_image_ph), reduction_indices=[1]))
        L1 = tf.reduce_mean(tf.reduce_sum(tf.abs(t_org_image_ph - decoder_second_out), reduction_indices=[1]))
        kl_loss = tf.reduce_mean(-0.5 * tf.reduce_sum(1 + z_variance - tf.square(z_mean) - tf.exp(z_variance), 1))
        content_loss = tf.reduce_mean(tf.reduce_sum(tf.square(z - z1), reduction_indices=[1]))
        vae_loss = (10 * L2 + L1 + kl_loss + content_loss)
        # train_op = slim.train.AdamOptimizer(cfg.lr).minimize(vae_loss)
        global_step, train_step, lr,= dk.set_optimizer(lr_range=cfg.lr_range,num_batches_per_epoch=cfg.lr_decay_batches, loss=vae_loss)
        tf.summary.scalar('L2', L2)
        tf.summary.scalar('L1', L1)
        tf.summary.scalar('KL_Loss', kl_loss)
        tf.summary.scalar('VAE_Loss', vae_loss)
        tf.summary.scalar('Content_Loss', content_loss)

    # 求PSNR SSIM
    with tf.name_scope('PSNR_SSIM'):
        org_img, tran_img = ops.convert_type(t_org_image_ph, decoder_first_out)  # 训练的图片进行了归一化，现在将其转换成原始的图片格式
        org_img2, skip_img = ops.convert_type(t_org_image_ph, decoder_second_out)
        psnr_GR = tf.reduce_mean(tf.image.psnr(org_img, tran_img, 255))  # 求PSNR
        psnr_DR = tf.reduce_mean(tf.image.psnr(org_img, skip_img, 255))
        ssim_GR = tf.reduce_mean(tf.image.ssim(org_img, tran_img, 255))  # 求SSIM
        ssim_DR = tf.reduce_mean(tf.image.ssim(org_img, skip_img, 255))
        tf.summary.scalar('psnr_GR', psnr_GR)
        tf.summary.scalar('psnr_DR', psnr_DR)
        tf.summary.scalar('ssim_GR', ssim_GR)
        tf.summary.scalar('ssim_GR', ssim_DR)
    return t_image_ph, t_org_image_ph, decoder_first_out, decoder_second_out,psnr_GR, psnr_DR, ssim_GR, ssim_DR, global_step, train_step, lr,vae_loss
