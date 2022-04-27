import tensorflow as tf
import tensorflow.contrib.slim as slim
import os
import time
from choice import cfg_v2_1
from choice import mrvae_v2_1_2
from tools import utils
from tools import ops
from tools import PairDataSet_v2
from tools import development_kit as dk
from tools import visualize_fm_and_weight
from model.MSRnet import MSRnet

os.environ['CUDA_VISIBLE_DEVICES']='0'

cfg = cfg_v2_1
def train():

	# ---------------------------------------------------------------
	# 参数设计
	epochs_num = 8
	batchSize = 8
	model_path = 'E:\WuXu\Dawn\MRVAE-Modification\compare\MSRnet_test_x\low_gaussian_wx_linear_trans1_model2'
	if not os.path.exists(model_path): os.makedirs(model_path)
	msrnet = MSRnet()

	data = PairDataSet_v2.ListDataSet(path=r'E:\Dataset_LL', dataname='Synthetic_Lowlight_Dataset_wx',
									  pair=['high','low_gaussian_wx_linear_trans1'],
									  img_height=512, img_width=512)

	# ---------------------------------------------------------------
	# 输入ph
	t_image_ph = tf.placeholder(name='train_images', dtype=tf.float32,
	                            shape=[batchSize, cfg.image_size, cfg.image_size, 3])
	t_org_image_ph = tf.placeholder(name='train_images', dtype=tf.float32,
	                                shape=[batchSize, cfg.image_size, cfg.image_size, 3])

	# ---------------------------------------------------------------
	# 训练
	msrnet_out = msrnet.build_MSRnet(xImgs=t_image_ph, yImgs=t_org_image_ph)
	n_batch_train = int(data.train_num // cfg.batch_size)  # 每批次训练的次数
	ops.show_parament_numbers()  # 计算模型参数

	model_saver = tf.train.Saver(max_to_keep=3)
	# summary_merged = tf.summary.merge_all()
	with tf.Session() as sess:
		coord, threads = dk.init_variables_and_start_thread(sess)
		if cfg.restore_model:
			model_saver.restore(sess, tf.train.latest_checkpoint(cfg.ckpt))
			print("已加载模型 {}  ........".format(tf.train.latest_checkpoint(cfg.ckpt)))
		writer = tf.summary.FileWriter(cfg.train_summary_path, tf.get_default_graph())
		for ep in range(epochs_num):
			for step in range(n_batch_train):
				img_an, image_org = data.GetNextBatch(batchSize)
				l,_ = sess.run([msrnet_out.total_loss, msrnet_out.train_step],
												   feed_dict={t_image_ph: img_an, t_org_image_ph: image_org})
				if step % 10 == 0:
					print('[Training MSRnet] epoch:[{}/{}]     step:{}     loss:{:.4f}'.format(ep,epochs_num,step,l))
			model_name = str(ep) + '.ckpt'
			model_saver.save(sess, os.path.join(model_path, model_name))
		writer.close()
		dk.stop_threads(coord, threads)


if __name__ == '__main__':
	train()