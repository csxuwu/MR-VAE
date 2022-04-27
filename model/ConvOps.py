import tensorflow as tf
import tensorflow.contrib.slim as slim
from tflearn.layers.conv import global_avg_pool
from tensorflow.contrib.layers import batch_norm
from tensorflow.contrib.framework import arg_scope
import numpy as np

'''
    SE、ResNeXt、SE_ResNeXt卷积操作
'''

class SE_Unit():
    def __init__(self,num_outputs,kernel_size,stride,ratio=16):
        self.num_outputs = num_outputs
        self.kernel_size = kernel_size
        self.stride = stride
        self.ratio = ratio          # SE 中FC的缩放比例


    def _conv_tr(self, inputs, num_outputs, kernel_size=[3,3], stride=1, padding='same', scope='conv'):
        '''
        普通卷积操作
        :param inputs:
        :param num_outputs:
        :param kernel_size:
        :param stride:
        :return:
        '''
        with tf.name_scope(scope):
            output = slim.conv2d(inputs=inputs,num_outputs=num_outputs,kernel_size=kernel_size,stride=stride,padding=padding)
            return output


    def _global_average_pooling(self, input, scope='global_average_pooling'):
        '''
        全局平均池化
        :param input:
        :return:
        '''
        with tf.name_scope(scope):
            output = global_avg_pool(input,name='Global_avg_pooling')
            return output


    def _fc(self, inputs, ratio, num_outputs, scope='fc'):
        '''
        全连接操作：两个fc，第一个按比例缩小输出节点数，第二个恢复正常的输出节点数
        :param inputs:
        :param ratio:第一个fc输出节点数缩小比例
        :param num_outputs:
        :return:
        '''
        with tf.name_scope(scope):
            num_outputs_1 = num_outputs // ratio
            fc_1 = slim.fully_connected(inputs=inputs,num_outputs=num_outputs_1,activation_fn=None)
            relu = tf.nn.relu(fc_1)
            fc_2 = slim.fully_connected(inputs=relu,num_outputs=num_outputs,activation_fn=None)
            sigmoid = tf.nn.sigmoid(fc_2)
            output = tf.reshape(sigmoid,[-1,1,1,num_outputs])
            return output


    def Bulid_SE_Unit(self,input,is_only_SE=True,scope='SE_Unit'):
        with tf.name_scope(scope):
            if is_only_SE:
                # 如果是仅构建SE_Unit，则使用普通conv；否则，不使用普通conv，因为input已经经过某些conv处理了，如resnext_unit
                conv = self._conv_tr(inputs=input,num_outputs=self.num_outputs, kernel_size=self.kernel_size, stride=self.stride, scope=scope + "_conv_tr")
            else:
                conv = input
            global_avg_pool = self._global_average_pooling(conv, scope=scope + '_global_avg_pool')
            fc = self._fc(inputs=global_avg_pool, ratio=self.ratio, num_outputs=self.num_outputs, scope=scope + '_fc')
            SEUnit_out = input * fc
            return SEUnit_out


class ResNeXt_Unit_and_SE_ResNeXt_Unit(SE_Unit):
    def __init__(self,num_outputs,transform_groups,transform_groups_num_outputs,kernel_size,stride):
        SE_Unit.__init__(self,num_outputs=num_outputs,kernel_size=3,stride=1,ratio=16)
        self.num_outputs = num_outputs
        self.transform_groups = transform_groups
        self.transform_groups_num_outputs = transform_groups_num_outputs
        self.kernel_size = kernel_size
        self.stride = stride
        self.SE_Unit = SE_Unit(num_outputs,kernel_size=3,stride=1,ratio=16)

    def _concatenate(self,input):
        '''
        将split，transform的结果concat返回
        :param input:
        :return:
        '''
        output = tf.concat(input,axis=3)
        return output


    def _batch_norm(self,input,scope):
        '''
        批归一化
        :param input:
        :param training:是否训练
        :param scope:
        :return:
        '''
        with slim.arg_scope([slim.batch_norm],
                            scope=scope,
                            updates_collections=True,
                            decay=0.9,
                            center=True,
                            scale=True,
                            zero_debias_moving_mean=True):
            # if bool(training):
            #     return slim.batch_norm(inputs=input, is_training=training, reuse=None)
            # else:
            #     return slim.batch_norm(inputs=input, is_training=training,reuse=True)
            # print('batch:',type(training))
            # return tf.cond(training,lambda : slim.batch_norm(inputs=input,is_training=training,reuse=None),     # tf.cond相当于三元运算符，A > B ? C : D
            #                         lambda: slim.batch_norm(inputs=input, is_training=training, reuse=True))

            print(type(input))
            return batch_norm(inputs=input)

    def _split_transform(self,input,transform_groups_num_outputs,kernel_size,stride,scope):
        '''
        对输入进行split，transform
        :param input:
        :param transform_groups_num_outputs:
        :param kernel_size:
        :param stride:
        :param scope:
        :return:
        '''
        with tf.name_scope(scope):
            '''使用1x1 conv 对input进行split'''
            conv1 = slim.conv2d(inputs=input,num_outputs=transform_groups_num_outputs,kernel_size=[1,1],stride=1,activation_fn=None,scope=scope + '_split')
            # conv1 = self._batch_norm(conv1,self.training,scope + '_batch1')
            conv1 = slim.batch_norm(conv1, scope=scope + '_batch1')
            conv1 = tf.nn.relu(conv1)

            '''对split结果进行transform'''
            conv2 = slim.conv2d(inputs=conv1,num_outputs=transform_groups_num_outputs,kernel_size=kernel_size,stride=stride,activation_fn=None,scope=scope + '_transform')
            # conv2 = self._batch_norm(conv2,scope + '_batch2')
            conv2 = slim.batch_norm(conv2,scope=scope + '_batch2')
            conv2 = tf.nn.relu(conv2)

            return conv2


    def _split_transform_block(self,input,transform_groups,transform_groups_num_outputs,kernel_size=3,stride=3,scope='split_transform_block'):
        '''
        分组对input进行split，transform
        :param input:
        :param transform_groups:将input分成的group数量
        :param transform_groups_num_outputs:每个group的输出通道数
        :param kernel_size:transform的卷积核大小
        :param stride:transform的卷积核步长
        :return:
        '''
        with tf.name_scope(scope):
            split_out = list()
            for i in range(transform_groups):
                transform_out = self._split_transform(input,transform_groups_num_outputs,kernel_size,stride,scope=scope + '_group_' + str(i))
                split_out.append(transform_out)
            return self._concatenate(split_out)


    def _aggregating(self,input,num_outputs,scope):
        '''
        组合输出
        :param input:split+transform的输出
        :param num_outputs:
        :param scope:
        :return:
        '''
        with tf.name_scope(scope):
            conv = slim.conv2d(inputs=input,num_outputs=num_outputs,kernel_size=1,stride=1,activation_fn=None,scope=scope + '_conv')
            # batch = self._batch_norm(conv,training=self.training,scope=scope + '_batch')
            batch = slim.batch_norm(conv,scope=scope + '_batch')

            return batch


    def Build_ResNeXt_Unit(self,input,is_only_ResNeXt=True,scope='ResNeXt_Unit'):
        '''
        搭建ResNeXt_Unit
        :param input:
        :return:
        '''
        split_transform = self._split_transform_block(input,self.transform_groups,self.transform_groups_num_outputs,self.kernel_size,self.stride,scope=scope + '_split_transform_block')
        aggregating = self._aggregating(split_transform,self.num_outputs,scope=scope + 'aggregating')

        if is_only_ResNeXt:
            # 如果仅构建ResNeXt，则添加跳跃结构；否则，不需要
            output = tf.nn.relu(input + aggregating)
        else:
            output = aggregating
        return output


    def Build_SE_ResNeXt_Unit(self,input,scope='Build_SE_ResNeXt_Unit'):
        '''
        搭建SE_ResNeXt_Unit
        :param input:
        :param scope:
        :return:
        '''
        # scope = scope + '_[Build_SE_ResNeXt_Unit]'
        with tf.name_scope(scope):
            ResNeXt_Unit_output = self.Build_ResNeXt_Unit(input,is_only_ResNeXt=False,scope=scope + '_ResNeXt_Unit')
            SE_Unit_output = self.Bulid_SE_Unit(input=ResNeXt_Unit_output,is_only_SE=False,scope=scope + '_SE_Unit')
            output_channels = SE_Unit_output.get_shape()[-1]
            input = slim.conv2d(inputs=SE_Unit_output,num_outputs=output_channels,kernel_size=1,stride=1,activation_fn=None)
            output = tf.nn.relu(input + SE_Unit_output)
            return output