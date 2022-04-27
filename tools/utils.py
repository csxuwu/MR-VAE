import tensorflow as tf
import tensorflow.contrib.slim as slim

'''
    MRVAE中的各种操作：
        sampling：采样
        prelu_tf：激活函数
'''

'''激活函数'''
def prelu_tf(inputs, name='Prelu'):
    with tf.name_scope(name):
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            alphas = tf.get_variable('alpha', inputs.get_shape()[-1], initializer=tf.zeros_initializer(),
                                     dtype=tf.float32)
        pos = tf.nn.relu(inputs)
        neg = alphas * (inputs - abs(inputs)) * 0.5

        return pos + neg


'''子像素卷积，放大率2'''
def phaseShift(inputs, scale, shape_1, shape_2):
        # Tackle the condition when the batch is None
        X = tf.reshape(inputs, shape_1)
        X = tf.transpose(X, [0, 1, 3, 2, 4])
        return tf.reshape(X, shape_2)

def pixel_shuffler(inputs,scale):
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
            output = tf.concat([phaseShift(x, scale, shape_1, shape_2) for x in input_split], axis=3)
            # print('pixel_shuffler')
            # print('pixel_shuffler output:{}'.format(output.get_shape()))
            return output

def sampling(z_mean, z_variance):
    with tf.name_scope('sampling'):
        epsilon = tf.random_normal(shape=z_mean.get_shape(), mean=0, stddev=1)
        print(epsilon.get_shape())
        sampling = z_mean + tf.multiply(epsilon, tf.exp(z_variance / 2), name='sampling')
        return sampling
