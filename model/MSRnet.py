import tensorflow as tf
import tensorflow.contrib.slim as slim
import collections
from tools import ops
from tools import development_kit as dk

OUTPUT_DIR = '../aLLE_outputData'

class MSRnet:
    def __init__(self, ):
        self.lr_range = (1e-3,1e-6,0.96)
        self.lr_decay_batches = 5000  # 学习率每5000次衰减一次

    def build_MSRnet(self, xImgs, yImgs, name='MSRnet'):
        print(" ------------ build MSRnet ----------")
        MSRnet = collections.namedtuple('MSRnet', 'MSRnet_outImgs, mse_loss, L2_rl, total_loss,train_step')
        with tf.name_scope(name):
            MSRnet_outImgs = self.MSRnet_model(xImgs)
            weight_set = slim.get_trainable_variables(scope='MSRnet_model', suffix='weights')
            mse_loss, L2_rl, total_loss = self.MSRnet_Loss(MSRnet_outImgs, yImgs, weight_set)
            global_step, train_step, lr, = dk.set_optimizer(lr_range=self.lr_range,
                                                            num_batches_per_epoch=self.lr_decay_batches, loss=total_loss)
        return MSRnet(
            MSRnet_outImgs=MSRnet_outImgs,
            mse_loss=mse_loss,
            L2_rl=L2_rl,
            total_loss=total_loss,
	        train_step=train_step
        )

    def build_MSRnet_test(self, xImgs, name='build_MSRnet_test'):
        print(" ------------ build build_MSRnet_test ----------")
        with tf.name_scope(name):
            MSRnet_outImgs = self.MSRnet_model(xImgs)
            return MSRnet_outImgs


    def MSRnet_model(self, inputs, scope='MSRnet_model'):
        with tf.variable_scope(scope):
            # Multi-scale logarithmic transformation
            X1 = logTransform(inputs)
            X1 = slim.conv2d(X1, num_outputs=3, kernel_size=3, stride=1, activation_fn=None)

            # Difference of convolution
            HK10 = ConvNet(X1,kernels=[3,3,3,3,3,3,3,3,3,3],
                              outputs=[32,32,32,32,32,32,32,32,32,32])

            # color restoration function
            X2 = X1 - HK10
            Y_hat = slim.conv2d(X2, num_outputs=3, kernel_size=1, stride=1, activation_fn=None)

            # 归一化运算
            Y_hat = tf.divide(tf.subtract(Y_hat, tf.reduce_min(Y_hat)),
                              tf.subtract(tf.reduce_max(Y_hat), tf.reduce_min(Y_hat)))
        return Y_hat

    def MSRnet_Loss(self, xImgs, yImgs, weight_set, scope='MSRnet_Loss'):
        with tf.variable_scope(scope):
            mse_loss = tf.reduce_mean(tf.square(tf.subtract(yImgs, xImgs)))

            # L2正则化
            L2_rl = tf.add_n([tf.nn.l2_loss(v) for v in weight_set]) * 0.001

            total_loss = mse_loss + L2_rl
        return mse_loss, L2_rl, total_loss


def logTransform(input,scale=[1.0, 10.0, 100.0, 300.0], toO_1=False, conv_to_3_channel=True):
    '''
    计算对数变换
    :param input: 一个4-D tensor，[NHWC]
    :param scale: 变换的尺度列表
    :return: 对数变换之后的张量 [NHW  n*C ]，其中，n是scale个数
    '''
    m=[]
    for i in range(len(scale)):
        temp=tf.divide(tf.log(tf.add(input,scale[i])),tf.log(tf.add(scale[i],1.0)))
        if toO_1:
            # 规范化到[0,1]
            temp=tf.divide(tf.subtract(temp,tf.reduce_min(temp)),tf.subtract(tf.reduce_max(temp),tf.reduce_min(temp)))
        m.append(temp)
    res=tf.concat(m,axis=3,name='L_Concat')

    if(conv_to_3_channel):
    #利用1x1卷积，变成3通道
        res=slim.conv2d(res, num_outputs=32,kernel_size=1,stride=1,activation_fn=tf.nn.relu)
    return res


def ConvNet(input,kernels,outputs):
    out=[]
    for i in range(len(kernels)):
        net=slim.conv2d(input,num_outputs=outputs[i],kernel_size=kernels[i],padding='SAME',activation_fn=tf.nn.relu)
        out.append(net)
    out=tf.concat([out[i] for i in range(len(out))],axis=3,name='H_concat')
    # Hk_1=slim.conv2d(out,num_outputs=3,kernel_size=1,activation_fn=tf.nn.relu)
    Hk_1=slim.conv2d(out,num_outputs=3, kernel_size=1, stride=1, activation_fn=None)
    return Hk_1

