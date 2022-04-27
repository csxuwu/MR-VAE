import tensorflow as tf
import tensorflow.contrib.slim as slim
import os
from tools import ops, PairDataSet_v2
import time

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
        第一阶段：
            网络结构概述：
                在multi_stage_vae 的基础上添加了U-Net ，将encoder的信息跳跃送到decoder的第一阶段，使用concat添加
                每层解码、编码的设计参考了U-Net，3层卷积 3x3 s=1，加一个池化来做下采样
            输入数据：低照度图像
            损失函数：L2  
            上采样操作：resize+conv
            激活函数：relu
            输出处理：sigmoid
        第二阶段：
            网络结构概述：
                参考了SRResnet主体结构,使用3个残差快    
                将第一阶段编码的第一层的特征输出传递到第二阶段，作为输出的一部分，以concat连接 
            输入数据：第一阶段经过sigmoid处理的输出
            损失函数：L1
            激活函数：prelu
            输出处理：sigmoid
        **调试：
            改变采样规则 
            计算正常照度与低照度图像编码阶段的L2损失，中间输出层不计算，只计算结果
            添加SENet
            
'''


class multi_vae():
    def __init__(self, name, latent_dim, image_size, epoch, batch_size, lr, lr_decay, train_summary_path,train_out_path,test_summary_path,test_out_path,train_dataset_path,test_dataset_path):
        self.name = name
        self.latent_dim = latent_dim
        self.image_size = image_size
        self.epoch = epoch
        self.batch_size = batch_size
        self.lr = lr
        self.lr_decay = lr_decay

        self.train_summary_path = train_summary_path
        self.train_out_path = train_out_path
        self.test_summary_path = test_summary_path
        self.test_out_path = test_out_path
        self.train_dataset_path = train_dataset_path
        self.test_dataset_path = test_dataset_path

    def visual_image(self, name, input):

        input_image = tf.reshape(tensor=input, shape=[-1, self.image_size, self.image_size, 3])
        tf.summary.image(name=name, tensor=input_image, max_outputs=4)

    def sampling(self, z_mean, z_variance):
        with tf.name_scope('sampling'):
            epsilon = tf.random_normal(shape=z_mean.get_shape(),mean=0,stddev=1)
            print(epsilon.get_shape())
            sampling = z_mean + tf.multiply(epsilon, tf.exp(z_variance/2), name='sampling')
            return sampling

    def prelu_tf(self, inputs, name='Prelu'):
        with tf.name_scope(name):
            with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
                alphas = tf.get_variable('alpha', inputs.get_shape()[-1], initializer=tf.zeros_initializer(),
                                         dtype=tf.float32)
            pos = tf.nn.relu(inputs)
            neg = alphas * (inputs - abs(inputs)) * 0.5

            return pos + neg

    '''解码'''
    '''子像素卷积，放大率2'''
    def phaseShift(self,inputs, scale, shape_1, shape_2):
        # Tackle the condition when the batch is None
        X = tf.reshape(inputs, shape_1)
        X = tf.transpose(X, [0, 1, 3, 2, 4])
        return tf.reshape(X, shape_2)

    def pixel_shuffler(self,inputs,scale):
        with tf.name_scope('pixel_shuffler'):
            size = tf.shape(inputs)
            batch_size = size[0]
            h = size[1]
            w = size[2]
            c = inputs.get_shape().as_list()[-1]

            # Get the target channel size
            channel_target = c // (scale * scale)
            channel_factor = c // channel_target

            shape_1 = [batch_size, h, w, channel_factor // scale, channel_factor // scale]
            shape_2 = [batch_size, h * scale, w * scale, 1]

            # Reshape and transpose for periodic shuffling for each channel
            input_split = tf.split(inputs, channel_target, axis=3)
            output = tf.concat([self.phaseShift(x, scale, shape_1, shape_2) for x in input_split], axis=3)
            # print('pixel_shuffler')
            # print('pixel_shuffler output:{}'.format(output.get_shape()))
            return output

    def encoder_block(self, name, inputs, num_outputs,is_pool=True):
        '''
        编码模块，返回池化之前的卷积输出，用于跳跃传递信息，和池化之后的输出，用于下一层编码
        :param name:
        :param inputs:
        :param num_outputs:
        :param is_pool:
        :return:
        '''
        with tf.name_scope(name):
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
                    conv1 = slim.conv2d(inputs=inputs, num_outputs=num_outputs)
                    conv2 = slim.conv2d(inputs=conv1, num_outputs=num_outputs)
                    if is_pool:
                        max_pool = slim.max_pool2d(conv2, scope=name)
                        print(name + ':{}'.format(conv2.get_shape()))
                        print(name + '_max_pool:{}'.format(max_pool.get_shape()))
                        return conv2,max_pool
                    else:
                        print(name + ':{}'.format(conv2.get_shape()))
                        return conv2

    def encoder(self, inputs):
        '''
        参考U-Net编码结构
        :param inputs:
        :return:
        '''
        with tf.name_scope('encoder'):
            print('encoder')
            '''返回池化前的卷积输出，作为跳跃链接到解码阶段；返回池化后的输出，作为下一层编码的输入'''
            conv1,max_pool1 = self.encoder_block(name='downsampling1', inputs=inputs, num_outputs=64)
            conv2,max_pool2 = self.encoder_block(name='downsampling2', inputs=max_pool1, num_outputs=128)
            conv3,max_pool3 = self.encoder_block(name='downsampling3', inputs=max_pool2, num_outputs=256)
            conv4,max_pool4 = self.encoder_block(name='downsampling4', inputs=max_pool3, num_outputs=512)
            conv51 = self.encoder_block(name='z_mean', inputs=max_pool4, num_outputs=self.latent_dim,is_pool=False)      # 最后一层，计算均值，不需要池化
            conv52 = self.encoder_block(name='z_variance', inputs=max_pool4, num_outputs=self.latent_dim,is_pool=False)  # 最后一层，计算方差，不需要池化
            return conv1, conv2, conv3,conv4,conv51,conv52

    def decoder_block(self, name, inputs, num_outputs, num_outputs_up=None, en_conv=None,is_transpose=False,is_pixel_shuffler=False,is_resize=False,img_width=512,img_height=512):
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
                    conv1 = slim.conv2d(inputs=inputs, num_outputs=num_outputs)
                    conv2 = slim.conv2d(inputs=conv1, num_outputs=num_outputs)
                    if is_transpose:            # 采用反卷积
                        t_conv1 = slim.conv2d_transpose(inputs=conv2, num_outputs=num_outputs_up, scope=name)
                        print(name + ':{}'.format(conv2.get_shape()))
                        print(name + '_transposel:{}'.format(t_conv1.get_shape()))
                        return t_conv1
                    elif is_resize:             # 采用resize+conv
                        r_conv = tf.image.resize_images(images=conv2,size=[img_width,img_height])
                        up_conv = slim.conv2d(inputs=r_conv,num_outputs=num_outputs_up,scope=name)
                        print(name + ':{}'.format(conv2.get_shape()))
                        print(name + '_resize:{}'.format(up_conv.get_shape()))
                        return up_conv
                    elif is_pixel_shuffler:     # 采用子像素卷积
                        up_conv = self.pixel_shuffler(conv2, 2)
                        print(name + ':{}'.format(conv2.get_shape()))
                        print(name + '_pixel_shuuffler:{}'.format(up_conv.get_shape()))
                        return up_conv
                    else:                       # 不进行上采样
                        print(name + ':{}'.format(conv2.get_shape()))
                        return conv2

    def decoder_first(self, inputs, conv1, conv2, conv3, conv4):
        with tf.name_scope('decoder_first'):
            with slim.arg_scope([slim.conv2d],
                                kernel_size=3,
                                stride=1,
                                weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                padding='SAME',
                                activation_fn=tf.nn.relu):
                print('de')
                '''上采样'''
                upsampling1 = self.decoder_block(name='upsampling1', inputs=inputs, num_outputs=self.latent_dim,num_outputs_up=512,is_resize=True,img_width=64,img_height=64)
                upsampling2 = self.decoder_block(name='upsampling2', inputs=upsampling1, num_outputs=512,num_outputs_up=256,en_conv=conv4,is_resize=True,img_width=128,img_height=128)
                upsampling3 = self.decoder_block(name='upsampling3', inputs=upsampling2, num_outputs=256,num_outputs_up=128,en_conv=conv3,is_resize=True,img_width=256,img_height=256)
                upsampling4 = self.decoder_block(name='upsampling4', inputs=upsampling3, num_outputs=128,num_outputs_up=64,en_conv=conv2,is_resize=True,img_width=512,img_height=512)
                # upsampling1 = self.decoder_block(name='upsampling1', inputs=inputs, num_outputs=self.latent_dim,num_outputs_up=512,is_transpose =True)
                # upsampling2 = self.decoder_block(name='upsampling2', inputs=upsampling1, num_outputs=512,num_outputs_up=256,en_conv=conv4,is_transpose=True)
                # upsampling3 = self.decoder_block(name='upsampling3', inputs=upsampling2, num_outputs=256,num_outputs_up=128,en_conv=conv3,is_transpose=True)
                # upsampling4 = self.decoder_block(name='upsampling4', inputs=upsampling3, num_outputs=128,num_outputs_up=64,en_conv=conv2,is_transpose=True)
                # upsampling1 = self.decoder_block(name='upsampling1', inputs=inputs, num_outputs=self.latent_dim,num_outputs_up=512,is_pixel_shuffler =True)
                # upsampling2 = self.decoder_block(name='upsampling2', inputs=upsampling1, num_outputs=512,num_outputs_up=256,en_conv=conv4,is_pixel_shuffler=True)
                # upsampling3 = self.decoder_block(name='upsampling3', inputs=upsampling2, num_outputs=256,num_outputs_up=128,en_conv=conv3,is_pixel_shuffler=True)
                # upsampling4 = self.decoder_block(name='upsampling4', inputs=upsampling3, num_outputs=128,num_outputs_up=64,en_conv=conv2,is_pixel_shuffler=True)

                '''输出'''
                de_conv = self.decoder_block(name='de_conv', inputs=upsampling4, num_outputs=64,en_conv=conv1)
                de_out = slim.conv2d(inputs=de_conv,num_outputs=3,kernel_size=1, activation_fn=tf.nn.sigmoid)
                print('de_out:{}'.format(de_out.get_shape()))
                return de_out

    def residual_block(self, no, inputs, inputs64=None):
        '''
        构造残差模块
        :param no:
        :param inputs:
        :param inputs64:第一次残差与输入相加时需要调整输入的通道数
        :return:
        '''
        name = 'residual_block' + str(no)
        with tf.name_scope(name):
            with slim.arg_scope([slim.conv2d],
                                kernel_size=3,
                                stride=1,
                                weights_initializer=slim.xavier_initializer(),
                                padding='SAME',
                                activation_fn=None):  # BN后再使用激活函数
                '''第一个卷积模块：卷积 + BN + PReLU'''
                residual1_conv1 = slim.conv2d(inputs=inputs, num_outputs=64)
                residual1_conv1_BN = slim.batch_norm(residual1_conv1)
                residual1_conv1_PRelu = self.prelu_tf(residual1_conv1_BN)

                '''第二个卷积模块：卷积 + BN + 跳跃  无激活函数'''
                residual1_conv2 = slim.conv2d(inputs=residual1_conv1_PRelu, num_outputs=64)
                residual1_conv2_BN = slim.batch_norm(residual1_conv2)

                '''SENet：全局平均池化+FC+ReLU+FC+Sigmoid'''
                r = 16      # 降维比例，需要多次调试
                global_pooling = tf.nn.avg_pool(value=residual1_conv2_BN,ksize=[1,512,512,1],strides=[1,512,512,1],padding='SAME')
                FC1 = slim.fully_connected(inputs=global_pooling,num_outputs=4)     # num_outputs=64/16
                ReLU = tf.nn.relu(FC1)
                FC2 = slim.fully_connected(inputs=ReLU,num_outputs=64)
                Sigmoid = tf.nn.sigmoid(FC2)
                SENet_out = tf.multiply(Sigmoid,residual1_conv2_BN)
                if inputs64 != None:
                    residual1_out = tf.add(inputs64, SENet_out)  # 元素级别的相加，将输入与残差模块的输出相加
                else:
                    residual1_out = tf.add(inputs, SENet_out)  # 元素级别的相加，将输入与残差模块的输出相加
                print('{}st residuak:{}'.format(no, residual1_out.get_shape()))
                return residual1_out

    def decoder_second(self, inputs, de_conv1):
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
                print("skip")
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
                out64_prelu = self.prelu_tf(out64, name='out64_prelu')
                out67 = slim.conv2d(out64_prelu, num_outputs=inputs_concat_channels, kernel_size=1,
                                    scope='out67')  # 调整通道数
                out67_add = tf.add(inputs_concat, out67)  # 将跳跃的输入通过跳跃传递过来

                out32 = slim.conv2d(out67_add, num_outputs=32, scope='out32')
                out32_prelu = self.prelu_tf(out32, name='out32_prelu')
                out3 = slim.conv2d(out32_prelu, num_outputs=3, kernel_size=1, scope='out3')
                out3_sigmoid = tf.nn.sigmoid(out3, name='out3_sigmoid')
                print('decoder_second_out:{}'.format(out3_sigmoid.get_shape()))
                return out3_sigmoid

    def train_vae_multi(self):
        t_image_ph = tf.placeholder(name='train_images', dtype=tf.float32,
                                    shape=[self.batch_size, self.image_size, self.image_size, 3])
        t_org_image_ph = tf.placeholder(name='train_images', dtype=tf.float32,
                                        shape=[self.batch_size, self.image_size, self.image_size, 3])

        with tf.name_scope(self.name):
            conv1, conv2, conv3, conv4, z_mean, z_variance = self.encoder(t_image_ph)
            conv11,conv21,conv31,conv41,z_mean1,z_variance1 = self.encoder(t_org_image_ph)
            z = self.sampling(z_mean, z_variance)
            z1 = self.sampling(z_mean1,z_variance1)
            decoder_first_out = self.decoder_first(z, conv1, conv2, conv3, conv4)
            decoder_second_out = self.decoder_second(decoder_first_out, conv1)

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
            psnr_first = tf.reduce_mean(tf.image.psnr(org_img, tran_img, 255))  # 求PSNR
            psnr_second = tf.reduce_mean(tf.image.psnr(org_img, skip_img, 255))
            ssim_first = tf.reduce_mean(tf.image.ssim(org_img, tran_img, 255))  # 求SSIM
            ssim_second = tf.reduce_mean(tf.image.ssim(org_img, skip_img, 255))
        return t_image_ph, t_org_image_ph, decoder_first_out, decoder_second_out, \
               psnr_first, psnr_second, ssim_first, ssim_second, train_op, vae_loss

    def train(self, ):
        t_image_ph, t_org_image_ph, decoder_first_out, decoder_second_out, psnr_first, psnr_second, ssim_first, ssim_second, train_op, vae_loss = self.train_vae_multi()
        data = PairDataSet_v2.ListDataSet(path=self.train_dataset_path, dataname="train", img_height=512, img_width=512)
        test_data = PairDataSet_v2.ListDataSet(path=self.test_dataset_path, dataname="test", img_height=512,
                                               img_width=512)
        summary_merged = tf.summary.merge_all()
        model_saver = tf.train.Saver(max_to_keep=3)
        init = tf.global_variables_initializer()
        summary_path = 'D:\WuXu\Code\Python_code/vae_multi09/vae_multi0903\log/vae_multi0903-07/train_epoch6_2/summary'
        out_path = 'D:\WuXu\Code\Python_code/vae_multi09/vae_multi0903\log/vae_multi0903-07/train_epoch6_2/out'
        ops.create_file(summary_path)
        ops.create_file(out_path)
        name_str = self.name + '_train_epoch6_2_data.xlsx'
        excel_name = os.path.join(summary_path, name_str)
        excel, excel_active = ops.create_excel(excel_name)
        with tf.Session() as sess:
            sess.run(init)
            model_saver.restore(sess, tf.train.latest_checkpoint('D:\WuXu\Code\Python_code/vae_multi09/vae_multi0903\log/vae_multi0903-07/train_epoch5/summary'))
            writer = tf.summary.FileWriter(summary_path, tf.get_default_graph())
            global_steps = 0
            for ep in range(self.epoch):
                for step in range(data.train_num // self.batch_size):
                    img_an, image_org = data.GetNextBatch(self.batch_size)
                    _, summary = sess.run([train_op, summary_merged],
                                          feed_dict={t_image_ph: img_an, t_org_image_ph: image_org})

                    '''记录实验数据'''
                    if step % 100 == 0:
                        loss_value, psnr_transpose2, psnr_skip2, ssim_transpose2, ssim_skip2 = sess.run(
                            [vae_loss, psnr_first, psnr_second, ssim_first, ssim_second],
                            feed_dict={t_image_ph: img_an, t_org_image_ph: image_org})
                        '''模型数据'''
                        model_name = str(global_steps) + '.ckpt'
                        model_saver.save(sess, os.path.join(summary_path, model_name))
                        ops.data_output(excel_active, global_steps, step, loss_value, psnr_transpose2, psnr_skip2,
                                        ssim_transpose2, ssim_skip2, epoch, ep)
                        excel.save(excel_name)
                        print('已存储第{}global_steps的实验数据到excel:{}\n'.format(global_steps, excel_name))
                        '''图像数据'''
                        second_out = sess.run(decoder_second_out,
                                              feed_dict={t_image_ph: img_an, t_org_image_ph: image_org})
                        first_out = sess.run(decoder_first_out,
                                             feed_dict={t_image_ph: img_an, t_org_image_ph: image_org})
                        ops.img_save_for_all(out_path, global_steps, image_size, image_org, img_an, first_out,
                                             second_out)
                    writer.add_summary(summary, global_steps)
                    global_steps += 1
                # '''每个epoch测试一次'''
                # name_str2 = self.name + '_' + str(ep) + '_test_512valid.xlsx'
                # excel_name2 = os.path.join(self.test_summary_path, name_str2)
                # excel2, excel_activate2 = ops.create_excel(excel_name2)
                # l = data.train_num // self.batch_size
                # for step in range(l):
                #     img_an, img_org = test_data.GetNextBatch(self.batch_size)
                #     st = time.time()
                #     loss_value, summary, psnr_transpose2, psnr_skip2, ssim_transpose2, ssim_skip2 = sess.run(
                #         [vae_loss, summary_merged, psnr_first, psnr_second, ssim_first, ssim_second],
                #         feed_dict={t_image_ph: img_an, t_org_image_ph: img_org})
                #     # t[1,:,:,:]=img_an
                #
                #     '''记录实验数据'''
                #     # if (step + 1) % 10 == 0:
                #     '''模型数据'''
                #     test_epoch_name = str(ep) + 'epoch'
                #     test_out_path2 = os.path.join(self.test_out_path, test_epoch_name)
                #     ops.create_file(test_out_path2)
                #     Time = time.time() - st
                #     model_name2 = str(ep) + '.ckpt'
                #     model_saver.save(sess, os.path.join(test_out_path2, model_name2))
                #     ops.data_output(excel_activate2, l, step, loss_value, psnr_transpose2, psnr_skip2,
                #                     ssim_transpose2, ssim_skip2, Time)
                #     excel2.save(excel_name2)
                #     '''图像数据'''
                #     first_out = sess.run(decoder_second_out, feed_dict={t_image_ph: img_an, t_org_image_ph: img_org})
                #     second_out = sess.run(decoder_first_out,
                #                           feed_dict={t_image_ph: img_an, t_org_image_ph: img_org})
                #     ops.img_save(test_out_path2, l, image_size, img_org, img_an, second_out,
                #                  first_out, is_train=False)
            writer.close()

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

    def test(self):

        t_image_ph, t_org_image_ph, decoder_first_out, decoder_second_out, psnr_first, psnr_second, ssim_first, ssim_second, train_op, vae_loss = self.train_vae_multi()
        data = PairDataSet_v2.ListDataSet(path=self.test_dataset_path, dataname="test", img_height=512, img_width=512)
        model_saver = tf.train.Saver(max_to_keep=10)
        summary_merged = tf.summary.merge_all()
        summary_path =  'D:\WuXu\Code\Python_code/vae_multi09/vae_multi0903\log/vae_multi0903-07/test_epoch5/512valid/summary'
        out_path = 'D:\WuXu\Code\Python_code/vae_multi09/vae_multi0903\log/vae_multi0903-07/test_epoch5/512valid/out'
        ops.create_file(summary_path)
        ops.create_file(out_path)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())  # 清除默认图的堆栈，并设置全局图为默认图
            model_saver.restore(sess, tf.train.latest_checkpoint('D:\WuXu\Code\Python_code/vae_multi09/vae_multi0903\log/vae_multi0903-07/train_epoch5/summary'))
            # model_saver.restore(sess, '138600.ckpt')
            print('------ 已加载模型 ------')

            writer = tf.summary.FileWriter(summary_path, tf.get_default_graph())
            global_steps = 0
            name_str = self.name + '_test_ep5_512valid.xlsx'
            excel_name = os.path.join(summary_path, name_str)
            excel, excel_activate = ops.create_excel(excel_name)
            for step in range(data.train_num // self.batch_size):
                img_an, image_org = data.GetNextBatch(self.batch_size)
                st = time.time()
                loss_value, summary, psnr_transpose2, psnr_skip2, ssim_transpose2, ssim_skip2 = sess.run(
                    [vae_loss, summary_merged, psnr_first, psnr_second, ssim_first, ssim_second],
                    feed_dict={t_image_ph: img_an, t_org_image_ph: image_org})

                '''记录实验数据'''
                if (step + 1) % 1 == 0:
                    '''模型数据'''
                    Time = time.time() - st
                    # model_name = str(global_steps) + '.ckpt'
                    # model_saver.save(sess, os.path.join(self.test_summary_path, model_name))
                    ops.data_output(excel_activate, global_steps, step, loss_value, psnr_transpose2, psnr_skip2,
                                    ssim_transpose2, ssim_skip2, Time)
                    excel.save(excel_name)
                    '''图像数据'''
                    first_out = sess.run(decoder_second_out, feed_dict={t_image_ph: img_an, t_org_image_ph: image_org})
                    second_out = sess.run(decoder_first_out,
                                          feed_dict={t_image_ph: img_an, t_org_image_ph: image_org})
                    # test_out_path2 = os.path.join(self.test_out_path,str(epoch))
                    ops.img_save_for_all(out_path, global_steps, image_size, image_org, img_an, second_out,
                                         first_out, is_train=False)
                writer.add_summary(summary, global_steps)
                global_steps += 1
            writer.close()

if __name__ == '__main__':
    '''路径设置'''
    name = 'vae_multi0903-07'
    path = 'log/vae_multi0903-07'
    train_summary_path = os.path.join(path, 'train2\summary')
    train_out_path = os.path.join(path, 'train2/out')

    test_summary_path = os.path.join(path, 'test2/summary')
    # test_summary_path = 'D:\WuXu\Code\Python_code\Conpared_Experiment\log/vae_multi0903-04/'
    # test_out_path = 'D:\WuXu\Code\Python_code\Conpared_Experiment\log/vae_multi0903-04/'
    test_out_path = os.path.join(path, 'test2/out')

    # train_dataset_path = 'D:\WuXu\Code\Python_code\Lightlee_enhancement\data\gama_longexport'
    train_dataset_path = 'D:\WuXu\Code\Python_code\Lightlee_enhancement\data\gamatest512-train'
    test_dataset_path = 'D:\WuXu\Code\Python_code\Lightlee_enhancement\data\gamatest512-valid'
    # test_dataset_path = 'D:\WuXu\Code\Python_code\Lightlee_enhancement\data\dataset_2'
    # test_dataset_path = 'D:\WuXu\Code\Python_code\Lightlee_enhancement\data/realImages512'
    ops.create_file(train_summary_path)
    ops.create_file(test_summary_path)
    ops.create_file(train_out_path)
    ops.create_file(test_out_path)

    latent_dim = 1024
    image_size = 512
    epoch =1
    batch_size = 4
    lr = 0.001
    lr_decay = 0.99
    vae = multi_vae(name, latent_dim, image_size, epoch, batch_size, lr, lr_decay, train_summary_path, train_out_path,
                    test_summary_path, test_out_path, train_dataset_path, test_dataset_path)
    # vae.train_restore()
    vae.train()
    # vae.test()
    # vae.train_restore()

