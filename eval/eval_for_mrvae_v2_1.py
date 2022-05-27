
import tensorflow as tf
import os
import time
import matplotlib.pyplot as plt
from choice import cfg_v2_1
from choice import mrvae_v2_1_1
from choice import mrvae_v2_1_2
from tools import utils
from tools import ops
from tools import PairDataSet_v2
from tools import visualize_fm_and_weight
from tensorflow.python.framework import graph_util

# =========================== 参数 ===========================
cfg = cfg_v2_1
mrvae = mrvae_v2_1_2
# mrvae = mrvae_v2_1_1
# =========================== end ===========================

def eval():
	with tf.Graph().as_default() as graph:
	# t_image_ph, t_org_image_ph, decoder_first_out, decoder_second_out, psnr_first, psnr_second, ssim_first, ssim_second, train_op, vae_loss = mrvae()
		t_image_ph, t_org_image_ph, decoder_first_out, decoder_second_out, psnr_GR, psnr_DR, ssim_GR, ssim_DR, global_step, train_step, lr, vae_loss= mrvae() # v212

		# testing set
		# file_list = ['test_x','100EOS5D', 'DICM_640_480', 'dimImgs_960', 'ExDark120', 'LOLdataset_resize', 'nirscene1Dim2',
		#          'Phos2_0_8MP_R_resize', 'TID2013_dim', 'VVdataset_resize']
		file_list = ['temp']

		# FLOPs
		ops.stats_graph(graph)
		# file_list = ['fm']
		for file in (file_list):
			an_imgs_path = os.path.join(cfg.eval_dataset_path, file)
			tar_path = os.path.join(cfg.eval_out_path, file)
			ops.create_file(tar_path)
			# an_imgs = PairDataSet_v2.LoadImgFromPath(an_imgs_path)
			an_imgs, name_list = PairDataSet_v2.LoadImgFromPath2(an_imgs_path)

			total_run_time = 0.0
			model_saver = tf.train.Saver(max_to_keep=10)
			with tf.Session() as sess:
				sess.run(tf.global_variables_initializer())  # 清除默认图的堆栈，并设置全局图为默认图
				# model_saver.restore(sess, tf.train.latest_checkpoint(cfg.train_summary_path))
				# model_saver.restore(sess, tf.train.latest_checkpoint(r'E:\WuXu\Dawn\MRVAE-Modification\summary\mrvae_v2_1\mrvae_v2_1_2\Synthetic_Lowlight_Dataset_wx\summary'))
				model_saver.restore(sess, tf.train.latest_checkpoint(
					r'E:\WuXu\Dawn\MRVAE-Modification\summary\mrvae_v2_1\mrvae_v2_1_2\train_Synthetic_Lowlight_Dataset_wx\summary'))

				# model_saver.restore(sess, tf.train.latest_checkpoint(r'E:\WuXu\Dawn\MRVAE-Modification\summary\mrvae_v2_1\mrvae_v2_1_1\train\summary'))
				print('''==================== eval's model:{} ===================='''.format(cfg.name))
				print('==================== 当前测试数据集：{} ===================='.format(file))
				global_steps = 0
				name_str = cfg.name + '_' + file + '.xlsx'
				excel_name = os.path.join(tar_path, name_str)
				excel, excel_activate = ops.create_excel(excel_name)
				for step in range(len(an_imgs)):
					img_an0 = an_imgs[step]
					img_an = img_an0[tf.newaxis, :, :, :]

					# # =========================== 绘制特征图 ===========================
					# fm_names = ['encoder_conv1', 'encoder_conv2', 'encoder_conv3', 'encoder_conv4',
					#             'GR_upsampling1', 'GR_upsampling2', 'GR_upsampling3', 'GR_upsampling4',
					#             'DR_residual1', 'DR_residual2', 'DR_residual3', 'DR_residual4', 'DR_residual5',
					#             'DR_out64_prelu', 'DR_out32_prelu']
					# feature_maps_list = tf.get_collection('feature_maps')
					# fms = sess.run(tf.get_collection('feature_maps'),
					#                feed_dict={t_image_ph: img_an, t_org_image_ph: img_an})
					# # 一层conv的每个通道都绘制、每个通道叠加后得到一层conv的一张可视化图
					# visualize_fm_and_weight.visualize_fm_and_weight(feature_maps_list=feature_maps_list, fms=fms,
					#                                                 outpath=cfg.eval_out_path,
					#                                                 global_step=global_steps)
					# visualize_fm_and_weight.visualize_feature_map(fms)

					# # =========================== end ===========================

					'''记录实验数据'''
					'''模型数据'''
					st = time.time()
					DR = sess.run(decoder_second_out, feed_dict={t_image_ph: img_an})
					GR = sess.run(decoder_first_out, feed_dict={t_image_ph: img_an})
					Time = time.time() - st
					total_run_time += Time
					ops.data_output(excel_activate, global_steps, step, 0, 0, 0, 0, 0, Time)
					excel.save(excel_name)
					'''图像数据'''

					# rec_name_DR = str(step) + 'DR_' + cfg.name + '_.jpg'
					rec_name_DR = name_list[step] + cfg.name + '_.jpg'
					rec_name_GR = str(step) + 'GR_' + cfg.name + '_.jpg'
					# rec_name = str(step) + '.jpg'
					out_path_DR = os.path.join(tar_path, rec_name_DR)
					out_path_GR = os.path.join(tar_path, rec_name_GR)
					plt.imsave(out_path_DR, DR[0])
					# plt.imsave(out_path_GR, GR[0])
					global_steps += 1

				ave_run_time = total_run_time / float(len(an_imgs))
				print("[*] Average run time: %.4f" % ave_run_time)



def eval2():
	'''用于测试成对的合成图像'''
	# t_image_ph, t_org_image_ph, decoder_first_out, decoder_second_out, psnr_GR, psnr_DR, ssim_GR, ssim_DR, train_op, vae_loss = mrvae() # v 2_1_1
	t_image_ph, t_org_image_ph, decoder_first_out, decoder_second_out, psnr_GR, psnr_DR, ssim_GR, ssim_DR, global_step, train_step, lr, vae_loss = mrvae()  # v212
	# t_image_ph, t_org_image_ph, decoder_first_out, decoder_second_out, psnr_GR, psnr_DR, ssim_GR, ssim_DR, global_step, train_step, lr, vae_loss= mrvae()
	# data = PairDataSet_v2.ListDataSet(path='D:\WuXu\Code\Python_code\Lightlee_enhancement\data\dataset_2', dataname="test", img_height=512, img_width=512)
	test_set = 'test_low_gaussian_wx_linear_trans1_200'
	data = PairDataSet_v2.ListDataSet(path='E:\Dataset_LL\Synthetic_Lowlight_Dataset_wx',
									  dataname=test_set, img_height=512, img_width=512)
	total_run_time = 0.0
	tar_path = os.path.join(cfg.eval_out_path,test_set)
	ops.create_file(tar_path)

	model_saver = tf.train.Saver(max_to_keep=3)
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())  # 清除默认图的堆栈，并设置全局图为默认图
		model_saver.restore(sess, tf.train.latest_checkpoint(cfg.train_summary_path))
		print('''==================== eval's model:{} ===================='''.format(cfg.name))
		print('==================== 当前测试数据集：{} ===================='.format(test_set))
		global_steps = 0
		name_str = cfg.name + '_' + test_set +'.xlsx'
		excel_name = os.path.join(tar_path, name_str)
		excel, excel_activate = ops.create_excel(excel_name)

		# t = 234 / data.train_num
		
		for step in range(data.train_num // 1):
			img_an, image_org = data.GetNextBatch(1)
			'''记录实验数据'''
			psnr_GR2, psnr_DR2, ssim_GR2, ssim_DR2 = sess.run([psnr_GR, psnr_DR, ssim_GR, ssim_DR],feed_dict={t_image_ph: img_an, t_org_image_ph: image_org})
			st = time.time()
			out_img = sess.run(decoder_second_out, feed_dict={t_image_ph: img_an})
			Time = time.time() - st
			total_run_time += Time

			ops.data_output(excel_activate, global_steps, step, 0,psnr_first=psnr_GR2,psnr_second=psnr_DR2,ssim_first=ssim_GR2,ssim_second=ssim_DR2, Time=Time)
			excel.save(excel_name)

			'''图像数据'''
			rec_name = str(step) + '_' + cfg.name + '_.jpg'
			out_path = os.path.join(tar_path, rec_name)
			plt.imsave(out_path, out_img[0])
			global_steps += 1

		# ave_run_time = total_run_time / float(len(data.train_num))
		ave_run_time = total_run_time / data.train_num
		print("[*] Average run time: %.4f" % ave_run_time)

if __name__ == '__main__':
	eval()


