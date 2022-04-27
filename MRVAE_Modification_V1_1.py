import tensorflow as tf
import tensorflow.contrib.slim as slim
import os
import matplotlib.pyplot as plt
from tools import ops, utils, PairDataSet_v2
import time
from model.ConvOps import ResNeXt_Unit_and_SE_ResNeXt_Unit
from choice import cfg

os.environ['CUDA_VISIBLE_DEVICES'] = "1"
'''
    低照度还原
        训练数据集：gamatest512-train
        超级参数设置：
            epoch = 1000
            batch_size = 4
            lr = 0.001
            lr_decay = 0.99
            优化器：Adam
        encoder:
            5 个 SE_ResNeXt卷积，每个卷积操作之间加池化操作
        GR：
            5 个 SE_ResNeXt卷积，每个卷积操作之间加resize+conv，作为上采样操作
        DR：
            5 个 残差块，每个残差块的卷积使用SE_ResNeXt卷积
'''


class MRVAE_Modification_V1():
    def __init__(self, name, latent_dim, image_size, epoch, batch_size, lr, lr_decay, train_summary_path,train_out_path,test_summary_path,test_out_path,train_dataset_path,test_dataset_path,is_training=True):
        self.name = name
        self.latent_dim = latent_dim
        self.image_size = image_size
        self.epoch = epoch
        self.batch_size = batch_size
        self.lr = lr
        self.lr_decay = lr_decay
        self.is_training = is_training

        self.train_summary_path = train_summary_path
        self.train_out_path = train_out_path
        self.test_summary_path = test_summary_path
        self.test_out_path = test_out_path
        self.train_dataset_path = train_dataset_path
        self.test_dataset_path = test_dataset_path


    '''编码'''
    def encoder_block(self, name, inputs, num_outputs,is_pool=True,scope=''):
        '''
        编码模块，返回池化之前的卷积输出，用于跳跃传递信息，和池化之后的输出，用于下一层编码
        :param name:
        :param inputs:
        :param num_outputs:
        :param is_pool:
        :return:
        '''
        # print('1',type(is_training))
        transform_groups = num_outputs // 8     # resnext 中间转换组数量 8 = 2 * 4     总的中间转换通道数是num_outputs的1/2，每个组的输出是通道是4，所以groups=num_outputs// (2*8)
        ResNeXt_conv = ResNeXt_Unit_and_SE_ResNeXt_Unit(num_outputs=num_outputs,transform_groups=transform_groups,transform_groups_num_outputs=4,kernel_size=3,stride=1)
        with tf.name_scope(name):
            with slim.arg_scope([slim.separable_conv2d],
                                kernel_size=3,
                                weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                padding='SAME',
                                activation_fn=tf.nn.relu):
                with slim.arg_scope([slim.max_pool2d],
                                    kernel_size=2,
                                    stride=2,
                                    padding='VALID',
                                    ):
                    '''卷积操作'''
                    # conv = ResNeXt_conv.Build_SE_ResNeXt_Unit(input=inputs,scope=scope + '_' + name + '_conv')
                    conv = slim.separable_conv2d(inputs=inputs,num_outputs=num_outputs,depth_multiplier=1,scope=scope + '_' + name + '_conv1')
                    conv = slim.separable_conv2d(inputs=conv,num_outputs=num_outputs,depth_multiplier=1,scope=scope + '_' + name + '_conv2')
                    '''下采样操作'''
                    if is_pool:
                        max_pool = slim.max_pool2d(conv, scope=scope + '_' + name + '_maxpooling')
                        print(name + ':{}'.format(conv.get_shape()))
                        print(name + '_max_pool:{}'.format(max_pool.get_shape()))
                        return conv,max_pool
                    else:
                        print(name + ':{}'.format(conv.get_shape()))
                        return conv

    def encoder(self, inputs,scope):
        '''
        参考U-Net编码结构
        :param inputs:
        :return:
        '''
        with tf.name_scope('encoder'):
            print('========== Encoder ==========')
            '''返回池化前的卷积输出，作为跳跃链接到解码阶段；返回池化后的输出，作为下一层编码的输入'''
            conv1,max_pool1 = self.encoder_block(name='downsampling1', inputs=inputs, num_outputs=64,scope=scope)
            conv2,max_pool2 = self.encoder_block(name='downsampling2', inputs=max_pool1, num_outputs=128,scope=scope)
            conv3,max_pool3 = self.encoder_block(name='downsampling3', inputs=max_pool2, num_outputs=256,scope=scope)
            conv4,max_pool4 = self.encoder_block(name='downsampling4', inputs=max_pool3, num_outputs=512,scope=scope)
            conv51 = self.encoder_block(name='z_mean', inputs=max_pool4, num_outputs=self.latent_dim,is_pool=False,scope=scope)      # 最后一层，计算均值，不需要池化
            conv52 = self.encoder_block(name='z_variance', inputs=max_pool4, num_outputs=self.latent_dim,is_pool=False,scope=scope)  # 最后一层，计算方差，不需要池化
            return conv1, conv2, conv3,conv4,conv51,conv52

    '''解码'''
    def decoder_block(self, name, inputs, num_outputs,num_outputs_up=None,en_conv=None,is_transpose=False,is_pixel_shuffler=False,is_resize=False,img_width=512,img_height=512,is_training=True):
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
        transform_groups = num_outputs // 8
        ResNeXt_conv = ResNeXt_Unit_and_SE_ResNeXt_Unit(num_outputs=num_outputs,transform_groups=transform_groups,transform_groups_num_outputs=4,kernel_size=3,stride=1)
        with tf.name_scope(name):
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
                    '''卷积操作'''
                    conv = slim.conv2d(inputs=inputs,num_outputs=num_outputs,scope=name + '_conv1')
                    conv = slim.conv2d(inputs=conv,num_outputs=num_outputs,scope=name + '_conv2')
                    '''上采样操作'''
                    if is_transpose:            # 采用反卷积
                        t_conv1 = slim.conv2d_transpose(inputs=conv, num_outputs=num_outputs_up, scope=name + '_transpose')
                        print(name + ':{}'.format(conv.get_shape()))
                        print(name + '_transposel:{}'.format(t_conv1.get_shape()))
                        return t_conv1
                    elif is_resize:             # 采用resize+conv
                        r_conv = tf.image.resize_images(images=conv,size=[img_width,img_height])
                        up_conv = slim.conv2d(inputs=r_conv,num_outputs=num_outputs_up,scope=name + '_resize')
                        print(name + ':{}'.format(conv.get_shape()))
                        print(name + '_resize:{}'.format(up_conv.get_shape()))
                        return up_conv
                    elif is_pixel_shuffler:     # 采用子像素卷积
                        up_conv = utils.pixel_shuffler(conv, 2)
                        print(name + ':{}'.format(conv.get_shape()))
                        print(name + '_pixel_shuuffler:{}'.format(up_conv.get_shape()))
                        return up_conv
                    else:                       # 不进行上采样
                        print(name + ':{}'.format(conv.get_shape()))
                        return conv

    '''GR 全局重构'''
    def GR(self, inputs, conv1, conv2, conv3, conv4, up_w, up_h):
        '''
        构建全局特征
        :param inputs:
        :param conv1:
        :param conv2:
        :param conv3:
        :param conv4:
        :return:
        '''
        with tf.name_scope('decoder_first'):
            with slim.arg_scope([slim.conv2d],
                                kernel_size=3,
                                stride=1,
                                weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                padding='SAME',
                                activation_fn=tf.nn.relu):
                print('========== GR ==========')
                '''上采样'''
                up_w *= 2;up_h *= 2
                upsampling1 = self.decoder_block(name='upsampling1', inputs=inputs, num_outputs=self.latent_dim,num_outputs_up=512,is_resize=True,img_width=up_w,img_height=up_h)
                up_w *= 2;up_h *= 2
                upsampling2 = self.decoder_block(name='upsampling2', inputs=upsampling1, num_outputs=512,num_outputs_up=256,en_conv=conv4,is_resize=True,img_width=up_w,img_height=up_h)
                up_w *= 2;up_h *= 2
                upsampling3 = self.decoder_block(name='upsampling3', inputs=upsampling2, num_outputs=256,num_outputs_up=128,en_conv=conv3,is_resize=True,img_width=up_w,img_height=up_h)
                up_w *= 2;up_h *= 2
                upsampling4 = self.decoder_block(name='upsampling4', inputs=upsampling3, num_outputs=128,num_outputs_up=64,en_conv=conv2,is_resize=True,img_width=up_w,img_height=up_h)

                '''输出'''
                de_conv = self.decoder_block(name='de_conv', inputs=upsampling4, num_outputs=64,en_conv=conv1)
                de_out = slim.conv2d(inputs=de_conv,num_outputs=3,kernel_size=1, activation_fn=tf.nn.sigmoid)
                print('de_out:{}'.format(de_out.get_shape()))
                return de_out

    '''DR'''
    def residual_block(self, no, inputs, inputs64=None):
        '''
        构造残差模块
        :param no:
        :param inputs:
        :param inputs64:第一次残差与输入相加时需要调整输入的通道数
        :return:
        '''
        ResNeXt = ResNeXt_Unit_and_SE_ResNeXt_Unit(num_outputs=64,transform_groups=8,transform_groups_num_outputs=4,kernel_size=3,stride=1)  # 残差块的输出通道是固定的
        name = 'residual_block' + str(no)
        with tf.name_scope(name):
            with slim.arg_scope([slim.conv2d],
                                kernel_size=3,
                                stride=1,
                                weights_initializer=slim.xavier_initializer(),
                                padding='SAME',
                                activation_fn=None):  # BN后再使用激活函数
                '''第一个卷积模块：卷积 + BN + PReLU'''
                residual1_conv1 = ResNeXt.Build_SE_ResNeXt_Unit(input=inputs,scope=name + '_residual_conv1')

                '''第二个卷积模块：卷积 + BN + 跳跃  无激活函数'''
                residual1_conv2 = ResNeXt.Build_SE_ResNeXt_Unit(input=residual1_conv1,scope=name + '_residual_conv2')
                if inputs64 != None:
                    residual1_out = tf.add(inputs64, residual1_conv2)  # 元素级别的相加，将输入与残差模块的输出相加
                else:
                    residual1_out = tf.add(inputs, residual1_conv2)  # 元素级别的相加，将输入与残差模块的输出相加
                print('{}st residuak:{}'.format(no, residual1_out.get_shape()))
                return residual1_out

    def DR(self, inputs, de_conv1, is_training=True):
        '''
        第二阶段，参考SRResnet
        :param inputs:
        :param de_conv1:将编码的第一层输出传递过来
        :return:
        '''
        with tf.name_scope('decoder_second'):
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
                residual1 = self.residual_block(1, inputs_concat, inputs64)
                residual2 = self.residual_block(2, residual1)
                residual3 = self.residual_block(3, residual2)
                residual4 = self.residual_block(4, residual3)
                residual5 = self.residual_block(5, residual4)

                '''普通卷积，输出'''
                '''普通卷积，输出'''
                out64 = slim.conv2d(residual5, num_outputs=64, scope='out64')
                out64_prelu = utils.prelu_tf(out64, name='out64_prelu')
                out67 = slim.conv2d(out64_prelu, num_outputs=inputs_concat_channels, kernel_size=1,
                                    scope='out67')  # 调整通道数
                out67_add = tf.add(inputs_concat, out67)  # 将跳跃的输入通过跳跃传递过来

                out32 = slim.conv2d(out67_add, num_outputs=32, scope='out32')
                out32_prelu = utils.prelu_tf(out32, name='out32_prelu')
                out3 = slim.conv2d(out32_prelu, num_outputs=3, kernel_size=1, scope='out3')
                out3_sigmoid = tf.nn.sigmoid(out3, name='out3_sigmoid')
                print('decoder_second_out:{}'.format(out3_sigmoid.get_shape()))
                return out3_sigmoid

    '''训练函数体'''
    def train_vae_multi(self):
        t_image_ph = tf.placeholder(name='train_images', dtype=tf.float32,
                                    shape=[self.batch_size, self.image_size, self.image_size, 3])
        t_org_image_ph = tf.placeholder(name='train_images', dtype=tf.float32,
                                        shape=[self.batch_size, self.image_size, self.image_size, 3])

        with tf.name_scope(self.name):
            conv1, conv2, conv3, conv4, z_mean, z_variance = self.encoder(t_image_ph,scope='an')
            conv11,conv21,conv31,conv41,z_mean1,z_variance1 = self.encoder(t_org_image_ph,scope='org')
            z = utils.sampling(z_mean, z_variance)
            z1 = utils.sampling(z_mean1, z_variance1)
            z_w = z1.get_shape()[1];z_h = z1.get_shape()[2]
            decoder_first_out = self.GR(z, conv1, conv2, conv3, conv4, z_w, z_h)
            decoder_second_out = self.DR(decoder_first_out, conv1)

        with tf.name_scope('Loss'):
            L2 = tf.reduce_mean(tf.reduce_sum(tf.square(decoder_first_out - t_org_image_ph), reduction_indices=[1]))
            # loss1 = tf.sqrt(tf.reduce_mean(tf.reduce_sum(tf.square(decoder_first_out - t_org_image_ph), reduction_indices=[1])))  # RMSE
            L1 = tf.reduce_mean(tf.reduce_sum(tf.abs(t_org_image_ph - decoder_second_out), reduction_indices=[1]))
            kl_loss = tf.reduce_mean(-0.5 * tf.reduce_sum(1 + z_variance - tf.square(z_mean) - tf.exp(z_variance), 1))
            content_loss = tf.reduce_mean(tf.reduce_sum(tf.square(z - z1), reduction_indices=[1]))
            vae_loss = (10*L2 + L1 + kl_loss + content_loss)
            # vae_loss = (10*loss1 + loss2 + kl_loss)
            train_op = slim.train.AdamOptimizer(self.lr).minimize(vae_loss)
            tf.summary.scalar('L2', L2)
            tf.summary.scalar('L1', L1)
            tf.summary.scalar('KL_Loss', kl_loss)
            tf.summary.scalar('VAE_Loss', vae_loss)
            tf.summary.scalar('Content_Loss', content_loss)

        '''求PSNR SSIM'''
        with tf.name_scope('PSNR_SSIM'):
            org_img, tran_img = ops.convert_type(t_org_image_ph, decoder_first_out)  # 训练的图片进行了归一化，现在将其转换成原始的图片格式
            org_img2, skip_img = ops.convert_type(t_org_image_ph, decoder_second_out)
            psnr_GR = tf.reduce_mean(tf.image.psnr(org_img, tran_img, 255))  # 求PSNR
            psnr_DR = tf.reduce_mean(tf.image.psnr(org_img, skip_img, 255))
            ssim_GR = tf.reduce_mean(tf.image.ssim(org_img, tran_img, 255))  # 求SSIM
            ssim_DR = tf.reduce_mean(tf.image.ssim(org_img, skip_img, 255))
            tf.summary.scalar('psnr_GR',psnr_GR)
            tf.summary.scalar('psnr_DR',psnr_DR)
            tf.summary.scalar('ssim_GR',ssim_GR)
            tf.summary.scalar('ssim_GR',ssim_DR)
        return t_image_ph, t_org_image_ph, decoder_first_out, decoder_second_out, \
               psnr_GR, psnr_DR, ssim_GR, ssim_DR, train_op, vae_loss

    '''开启训练'''
    def train(self, ):
        t_image_ph, t_org_image_ph, decoder_GR, decoder_DR, psnr_GR, psnr_DR, ssim_GR, ssim_DR, train_op, vae_loss = self.train_vae_multi()
        data = PairDataSet_v2.ListDataSet(path=self.train_dataset_path, dataname="train", img_height=512, img_width=512)
        summary_merged = tf.summary.merge_all()
        model_saver = tf.train.Saver(max_to_keep=3)
        init = tf.global_variables_initializer()

        name_str = self.name + '_train_data.xlsx'
        excel_name = os.path.join(self.train_summary_path, name_str)
        excel, excel_active = ops.create_excel(excel_name)
        ops.show_parament_numbers()

        with tf.Session() as sess:
            sess.run(init)
            writer = tf.summary.FileWriter(self.train_summary_path, tf.get_default_graph())
            global_steps = 0
            for ep in range(self.epoch):
                for step in range(data.train_num // self.batch_size):
                    img_an, image_org = data.GetNextBatch(self.batch_size)
                    _, summary = sess.run([train_op, summary_merged],
                                          feed_dict={t_image_ph: img_an, t_org_image_ph: image_org})

                    '''记录实验数据'''
                    if step % 1000 == 0:
                        loss_value, psnr_transpose2, psnr_skip2, ssim_transpose2, ssim_skip2 = sess.run(
                            [vae_loss, psnr_GR, psnr_DR, ssim_GR, ssim_DR],
                            feed_dict={t_image_ph: img_an, t_org_image_ph: image_org})
                        '''模型数据'''
                        model_name = str(global_steps) + '.ckpt'
                        model_saver.save(sess, os.path.join(self.train_summary_path, model_name))
                        ops.data_output(excel_active, global_steps, step, loss_value, psnr_transpose2, psnr_skip2,
                                        ssim_transpose2, ssim_skip2, epoch, ep)
                        excel.save(excel_name)
                        print('已存储第{}global_steps的实验数据到excel:{}\n'.format(global_steps, excel_name))
                        '''图像数据'''
                        second_out = sess.run(decoder_DR,
                                              feed_dict={t_image_ph: img_an, t_org_image_ph: image_org})
                        first_out = sess.run(decoder_GR,
                                             feed_dict={t_image_ph: img_an, t_org_image_ph: image_org})
                        ops.img_save_for_all(self.train_out_path, global_steps, image_size, image_org, img_an, first_out,
                                             second_out)
                    writer.add_summary(summary, global_steps)
                    global_steps += 1
                # self.epoch_test(t_image_ph=t_image_ph,epoch=ep,decoder_DR=decoder_DR)       # 每个epoch测试一次
            writer.close()

    def epoch_test(self,t_image_ph,epoch,decoder_DR):

        file_list = ['100EOS5D', 'DICM_640_480', 'dimImgs_960', 'ExDark120', 'LOLdataset_resize', 'nirscene1Dim2',
                     'Phos2_0_8MP_R_resize', 'TID2013_dim', 'VVdataset_resize']
        for file in (file_list):
            print('==================== 当前测试数据集：{} ===================='.format(file))
            an_imgs_path = os.path.join('D:\WuXu\Code\Python_code\Lightlee_enhancement\data\LLE_testImgs', file)
            tar_path = os.path.join(self.test_out_path, file)
            ops.create_file(tar_path)
            an_imgs = PairDataSet_v2.LoadImgFromPath(an_imgs_path)

            model_saver = tf.train.Saver(max_to_keep=10)
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                model_saver.restore(sess, tf.train.latest_checkpoint(self.train_summary_path))
                print('------ 已加载模型 ------')
                global_steps = 0
                name_str = self.name + '_' + epoch + '_' +  file + '.xlsx'
                excel_name = os.path.join(tar_path, name_str)
                excel, excel_activate = ops.create_excel(excel_name)
                for step in range(len(an_imgs)):
                    img_an0 = an_imgs[step]
                    img_an = img_an0[tf.newaxis, :, :, :]
                    st = time.time()

                    '''记录实验数据'''
                    '''模型数据'''
                    Time = time.time() - st
                    ops.data_output(excel_activate, global_steps, step, 0, 0, 0, 0, 0, Time)
                    excel.save(excel_name)
                    '''图像数据'''
                    out_img = sess.run(decoder_DR, feed_dict={t_image_ph: img_an,is_training:False})
                    rec_name = str(epoch) + '_' + str(step) + '_' + self.name + '.jpg'
                    out_path = os.path.join(tar_path, rec_name)
                    plt.imsave(out_path, out_img[0])
                    global_steps += 1

    '''重载训练'''
    def train_restore(self):
        t_image_ph, t_org_image_ph, decoder_first_out, decoder_second_out, psnr_first, psnr_second, ssim_first, ssim_second, train_op, vae_loss = self.train_vae_multi()
        data = PairDataSet_v2.ListDataSet(path=self.train_dataset_path, dataname="train", img_height=512, img_width=512)

        # an_imgs = PairDataSet_v2.LoadImgFromPath('D:\WuXu\Code\Python_code\Lightlee_enhancement\data/realImages512/test_x/')
        # org_imgs = PairDataSet_v2.LoadImgFromPath('D:\WuXu\Code\Python_code\Lightlee_enhancement\data/realImages512/test_y/')
        test_data = PairDataSet_v2.ListDataSet(path=self.test_dataset_path, dataname="test", img_height=512, img_width=512)
        # test_image_ph = tf.placeholder(name='train_images', dtype=tf.float32,
        #                             shape=[self.batch_size, self.image_size, self.image_size, 3])
        # test_org_image_ph = tf.placeholder(name='train_images', dtype=tf.float32,
        #                                 shape=[self.batch_size, self.image_size, self.image_size, 3])

        model_saver = tf.train.Saver(max_to_keep=10)
        summary_merged = tf.summary.merge_all()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())  # 清除默认图的堆栈，并设置全局图为默认图
            model_saver.restore(sess, tf.train.latest_checkpoint('D:\WuXu\Code\Python_code/vae_multi09/vae_multi0903\log/vae_multi0903-04/train_restore2\summary'))
            # model_saver.restore(sess, '138600.ckpt')
            print('------ train_restore 已加载模型 ------')
            writer = tf.summary.FileWriter(self.train_summary_path, tf.get_default_graph())
            global_steps = 0
            name_str = self.name + '_test_data.xlsx'
            excel_name = os.path.join(self.train_summary_path, name_str)
            excel, excel_activate = ops.create_excel(excel_name)
            for ep in range(self.epoch):
                '''每个一个epoch测试一遍'''
                name_str2 = self.name + '_' + str(ep) + '_test_512valid.xlsx'
                excel_name2 = os.path.join(self.test_summary_path, name_str2)
                excel2, excel_activate2 = ops.create_excel(excel_name2)

                # for s in range(len(an_imgs)):
                    # img_an0 = an_imgs[s]
                    # test_img_an = img_an0[tf.newaxis, :, :, :]
                    # img_org0 = org_imgs[s]
                    # test_img_org = img_org0[tf.newaxis, :, :, :]
                test_img_an,test_img_org = test_data.GetNextBatch(self.batch_size)
                st = time.time()
                loss_value, summary, psnr_transpose2, psnr_skip2, ssim_transpose2, ssim_skip2 = sess.run(
                    [vae_loss, summary_merged, psnr_first, psnr_second, ssim_first, ssim_second],
                    feed_dict={t_image_ph: test_img_an, t_org_image_ph: test_img_org})
                Time = time.time() - st
                ops.data_output(excel_activate2, epoch, ep, loss_value, psnr_transpose2, psnr_skip2,
                                ssim_transpose2, ssim_skip2, Time)
                excel2.save(excel_name2)
                '''图像数据'''
                first_out = sess.run(decoder_second_out,
                                     feed_dict={t_image_ph: test_img_an, t_org_image_ph: test_img_org})
                second_out = sess.run(decoder_first_out,
                                      feed_dict={t_image_ph: test_img_an, t_org_image_ph: test_img_org})
                test_epoch_name = str(ep) + 'epoch'
                test_out_path2 = os.path.join(self.test_out_path, test_epoch_name)
                ops.create_file(test_out_path2)
                ops.img_save_for_all(test_out_path2, ep, image_size, test_img_org, test_img_an, second_out,
                                     first_out, is_train=False)
                for step in range(data.train_num // self.batch_size):
                # for step in range(8600):
                    image_an, image_org = data.GetNextBatch(self.batch_size)
                    _, summary = sess.run([train_op, summary_merged],
                                          feed_dict={t_image_ph: image_an, t_org_image_ph: image_org})

                    '''记录实验数据'''
                    if step % 100 == 0:
                        loss_value, psnr_transpose2, psnr_skip2, ssim_transpose2, ssim_skip2 = sess.run([vae_loss, psnr_first, psnr_second, ssim_first, ssim_second],
                            feed_dict={t_image_ph: image_an, t_org_image_ph: image_org})
                        '''模型数据'''
                        model_name = str(global_steps) + '.ckpt'
                        model_saver.save(sess, os.path.join(self.train_summary_path, model_name))
                        ops.data_output(excel_activate, global_steps, step, loss_value, psnr_transpose2, psnr_skip2,
                                        ssim_transpose2, ssim_skip2, epoch, ep)
                        excel.save(excel_name)
                        print('已存储第{}global_steps的实验数据到excel:{}\n'.format(global_steps, excel_name))
                        '''图像数据'''
                        second_out = sess.run(decoder_second_out,
                                              feed_dict={t_image_ph: image_an, t_org_image_ph: image_org})
                        first_out = sess.run(decoder_first_out,
                                             feed_dict={t_image_ph: image_an, t_org_image_ph: image_org})
                        ops.img_save_for_all(self.train_out_path, global_steps, image_size, image_org, image_an, first_out,
                                             second_out)
                    global_steps += 1
                model_ep_path = os.path.join(self.train_summary_path, 'epoch')
                ops.create_file(model_ep_path)
                model_name = str(epoch) + '.ckpt'
                model_saver.save(sess, os.path.join(self.train_summary_path, model_name))


            writer.close()

    def eval_systhesis(self):
        t_image_ph, t_org_image_ph, decoder_first_out, decoder_second_out, psnr_first, psnr_second, ssim_first, ssim_second, train_op, vae_loss = self.train_vae_multi()
        data = PairDataSet_v2.ListDataSet(path=self.test_dataset_path, dataname="test", img_height=512, img_width=512)
        model_saver = tf.train.Saver(max_to_keep=10)
        summary_merged = tf.summary.merge_all()

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())  # 清除默认图的堆栈，并设置全局图为默认图
            model_saver.restore(sess, tf.train.latest_checkpoint(cfg.train_summary_path))
            print('------ 已加载模型 ------')
            writer = tf.summary.FileWriter(cfg.eval_summary_path, tf.get_default_graph())
            global_steps = 0
            name_str = cfg.name + '_' + cfg.eval_dataset + '.xlsx'
            excel_name = os.path.join(cfg.eval_summary_path, name_str)
            excel, excel_activate = ops.create_excel(excel_name)
            for step in range(data.train_num // cfg.batch_size):
                img_an, image_org = data.GetNextBatch(cfg.batch_size)
                st = time.time()
                loss_value, summary, psnr_transpose2, psnr_skip2, ssim_transpose2, ssim_skip2 = sess.run(
                    [vae_loss, summary_merged, psnr_first, psnr_second, ssim_first, ssim_second],
                    feed_dict={t_image_ph: img_an, t_org_image_ph: image_org})

                '''记录实验数据'''
                if (step + 1) % 1 == 0:
                    '''模型数据'''
                    Time = time.time() - st
                    ops.data_output(excel_activate, global_steps, step, loss_value, psnr_transpose2, psnr_skip2,
                                    ssim_transpose2, ssim_skip2, Time)
                    excel.save(excel_name)
                    '''图像数据'''
                    first_out = sess.run(decoder_second_out, feed_dict={t_image_ph: img_an, t_org_image_ph: image_org})
                    second_out = sess.run(decoder_first_out,
                                          feed_dict={t_image_ph: img_an, t_org_image_ph: image_org})
                    # test_out_path2 = os.path.join(cfg.eval_out_path, str(cfg.epoch))
                    ops.img_save_for_all(cfg.eval_out_path, global_steps, cfg.image_size, image_org, img_an, first_out,
                                         second_out, is_train=False)
                writer.add_summary(summary, global_steps)
                global_steps += 1
            writer.close()


    def eval_real(self):
        t_image_ph, t_org_image_ph, decoder_first_out, decoder_second_out, psnr_first, psnr_second, ssim_first, ssim_second, train_op, vae_loss = self.train_vae_multi()
        # D:\766QLL\dim2rgb_imgs\LLE_testImgs
        # file_list = ['realImages512', 'realImgs2_512']
        # file_list = ['t']
        file_list = ['100EOS5D', 'DICM_640_480', 'dimImgs_960', 'ExDark120', 'LOLdataset_resize', 'nirscene1Dim2',
                     'Phos2_0_8MP_R_resize', 'TID2013_dim', 'VVdataset_resize']
        for file in (file_list):
            print('==================== 当前测试数据集：{} ===================='.format(file))
            data = PairDataSet_v2.ListDataSet(path=cfg.eval_dataset_path, dataname="test", img_height=512, img_width=512)
            # an_imgs_file = file + '/test_x/'
            # org_imgs_file = file + '/test_y/'

            an_imgs_path = os.path.join('D:\WuXu\Code\Python_code\Lightlee_enhancement\data\LLE_testImgs', file)
            tar_path = os.path.join(r'E:\WuXu\Dawn\MRVAE-Modification\summary\MRVAE_Modification_V1.1\test\out', file)
            # an_imgs_path = os.path.join(r'D:\WuXu\Code\Python_code\Conpared_Experiment\temp_input', file)
            # tar_path = os.path.join('D:\WuXu\Code\Python_code\Conpared_Experiment\log/vae_multi0903-04-restore-test', file)

            ops.create_file(tar_path)
            an_imgs = PairDataSet_v2.LoadImgFromPath(an_imgs_path)

            model_saver = tf.train.Saver(max_to_keep=10)
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())  # 清除默认图的堆栈，并设置全局图为默认图
                model_saver.restore(sess, tf.train.latest_checkpoint(cfg.train_summary_path))
                print('------ 已加载模型 ------')
                global_steps = 0
                name_str = cfg.name + file + '.xlsx'
                excel_name = os.path.join(tar_path, name_str)
                excel, excel_activate = ops.create_excel(excel_name)
                for step in range(len(an_imgs)):
                    img_an0 = an_imgs[step]
                    img_an = img_an0[tf.newaxis, :, :, :]
                    st = time.time()

                    '''记录实验数据'''
                    '''模型数据'''
                    Time = time.time() - st
                    ops.data_output(excel_activate, global_steps, step, 0, 0, 0,
                                    0, 0, Time)
                    excel.save(excel_name)
                    '''图像数据'''
                    out_img = sess.run(decoder_second_out, feed_dict={t_image_ph: img_an})
                    rec_name = str(step) + '_' + cfg.name + '.jpg'
                    out_path = os.path.join(tar_path, rec_name)
                    plt.imsave(out_path, out_img[0])
                    global_steps += 1


if __name__ == '__main__':
    '''路径设置'''
    name = 'MRVAE_Modification_V1.1'
    path = 'summary/MRVAE_Modification_V1.1'
    train_summary_path = os.path.join(path, 'train/summary')
    train_out_path = os.path.join(path, 'train/out')
    test_summary_path = os.path.join(path, 'test/summary')
    test_out_path = os.path.join(path, 'test/out')
    train_dataset_path = 'D:\WuXu\Code\Python_code\Lightlee_enhancement\data\gamatest512-train'
    # test_dataset_path = 'D:\WuXu\Code\Python_code\Lightlee_enhancement\data/realImages512'
    test_dataset_path = 'D:\WuXu\Code\Python_code\Lightlee_enhancement\data\gamatest512-valid'
    ops.create_file(train_summary_path)
    ops.create_file(test_summary_path)
    ops.create_file(train_out_path)
    ops.create_file(test_out_path)

    latent_dim = 1024
    image_size = 512
    epoch = 6
    batch_size = 2
    lr = 0.001
    lr_decay = 0.99
    is_training = True
    mrvae = MRVAE_Modification_V1(name, latent_dim, image_size, epoch, batch_size, lr, lr_decay, train_summary_path, train_out_path,
                    test_summary_path, test_out_path, train_dataset_path, test_dataset_path,is_training=is_training)
    # vae.train_restore()
    # mrvae.train()
    mrvae.eval_systhesis()
    # vae.test()
    # vae.train_restore()

