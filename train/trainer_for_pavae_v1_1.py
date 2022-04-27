import tensorflow as tf
import tensorflow.contrib.slim as slim
import os
import time
from choice import cfg_pavae_v1_1 as cfg
# from choice import pavae_v1_1 as pavae
from choice import pavae_v1_1_1 as pavae
from tools import utils
from tools import ops
from tools import PairDataSet_v2
from tools import development_kit as dk
from tools import visualize_fm_and_weight


def train():
    print('========================================== training dataset:{} =========================================='.format(cfg.train_dataset))
    t_image_ph, t_org_image_ph, decoder_GR, decoder_DR, psnr_GR, psnr_DR, ssim_GR, ssim_DR, global_steps, train_step, lr, vae_loss = pavae()
    data = PairDataSet_v2.ListDataSet(path=cfg.train_dataset_path, dataname="train", img_height=512, img_width=512)

    n_batch_train = int(data.train_num // cfg.batch_size)  # 每批次训练的次数
    total_steps = n_batch_train * cfg.epoch  # 总的训练次数
    name_str = cfg.name + '_' + cfg.train_dataset + '.xlsx'
    excel_name = os.path.join(cfg.train_summary_path, name_str)
    excel, excel_active = ops.create_excel(excel_name)
    ops.show_parament_numbers()  # 计算模型参数

    model_saver = tf.train.Saver(max_to_keep=3)
    summary_merged = tf.summary.merge_all()
    with tf.Session() as sess:
        coord, threads = dk.init_variables_and_start_thread(sess)
        if cfg.restore_model:
            model_saver.restore(sess, tf.train.latest_checkpoint(cfg.ckpt))
            print("已加载模型 {}  ........".format(tf.train.latest_checkpoint(cfg.ckpt)))
        writer = tf.summary.FileWriter(cfg.train_summary_path, tf.get_default_graph())
        for ep in range(cfg.epoch):
            for step in range(n_batch_train):
                img_an, image_org = data.GetNextBatch(cfg.batch_size)
                _, global_step, summary = sess.run([train_step, global_steps, summary_merged],
                                                   feed_dict={t_image_ph: img_an, t_org_image_ph: image_org})

                # =========================== 记录实验数据  ===========================
                if global_step % 100 == 0:
                    loss_value, psnr_GR2, ssim_GR2, psnr_DR2, ssim_DR2, lr2 = sess.run([vae_loss, psnr_GR, ssim_GR,psnr_DR, ssim_DR, lr],
                                                                   feed_dict={t_image_ph: img_an,t_org_image_ph: image_org})
                    first_out = sess.run(decoder_GR, feed_dict={t_image_ph: img_an, t_org_image_ph: image_org})
                    st = time.time()
                    seconde_out = sess.run(decoder_DR, feed_dict={t_image_ph: img_an, t_org_image_ph: image_org})
                    run_time = time.time() - st
                    # 模型数据
                    model_name = str(global_step) + '.ckpt'
                    model_saver.save(sess, os.path.join(cfg.train_summary_path, model_name))
                    ops.data_output(excel_active=excel_active, global_steps=total_steps, step=global_step,
                                    loss_value=loss_value, psnr_first=psnr_GR2, psnr_second=psnr_DR2, ssim_first=ssim_GR2,
                                    ssim_second=ssim_DR2, Time=run_time, epoch=cfg.epoch, ep=ep, lr=lr2)
                    excel.save(excel_name)
                    print('已存储第 {} steps的实验数据到excel:{}\n'.format(global_step, excel_name))
                    # 图像数据
                    ops.img_save_for_all(out_path=cfg.train_out_path, global_steps=global_step,
                                        image_size=cfg.image_size, image_org=image_org, img_an=img_an,
                                        first_out=first_out,second_out=seconde_out)
                # =========================== end ===========================

                # =========================== 绘制特征图 ===========================
                # if global_step % 1000 == 0:
                #     fm_names = {'encoder_conv1', 'encoder_conv2', 'encoder_conv3', 'encoder_conv4',
                #                 'GR_upsampling1', 'GR_upsampling2', 'GR_upsampling3', 'GR_upsampling4',
                #                 'DR_residual1', 'DR_residual2', 'DR_residual3', 'DR_residual4', 'DR_residual5',
                #                 'DR_out64_prelu', 'DR_out32_prelu'}
                #     fms = sess.run(tf.get_collection('feature_maps'),
                #                    feed_dict={t_image_ph: img_an, t_org_image_ph: image_org})
                #     # 一层conv的每个通道都绘制、每个通道叠加后得到一层conv的一张可视化图
                #     visualize_fm_and_weight.visualize_fm_and_weight(fm_names=fm_names, fms=fms,
                #                                                     outpath=cfg.train_out_path,
                #                                                     global_step=global_step)

                # =========================== end ===========================
                writer.add_summary(summary, global_step)
        writer.close()
        dk.stop_threads(coord, threads)
