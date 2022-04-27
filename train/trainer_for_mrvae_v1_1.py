
import tensorflow as tf
import tensorflow.contrib.slim as slim
import os
import time
from choice import cfg_v1_1
from choice import mrvae_v1_1
from tools import utils
from tools import ops
from tools import PairDataSet_v2


def train():
    t_image_ph, t_org_image_ph, decoder_GR, decoder_DR, psnr_GR, psnr_DR, ssim_GR, ssim_DR, train_op, vae_loss = mrvae_v1_1()
    data = PairDataSet_v2.ListDataSet(path=cfg_v1_1.train_dataset_path, dataname="train", img_height=512, img_width=512)

    name_str = cfg_v1_1.name + '_' + cfg_v1_1.train_dataset + '.xlsx'
    excel_name = os.path.join(cfg_v1_1.train_summary_path, name_str)
    excel, excel_active = ops.create_excel(excel_name)
    ops.show_parament_numbers()     # 计算模型参数

    summary_merged = tf.summary.merge_all()
    model_saver = tf.train.Saver(max_to_keep=3)
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        writer = tf.summary.FileWriter(cfg_v1_1.train_summary_path, tf.get_default_graph())
        global_steps = 0
        for ep in range(cfg_v1_1.epoch):
            for step in range(data.train_num // cfg_v1_1.batch_size):
                img_an, image_org = data.GetNextBatch(cfg_v1_1.batch_size)
                _, summary = sess.run([train_op, summary_merged],
                                      feed_dict={t_image_ph: img_an, t_org_image_ph: image_org})

                # 记录实验数据
                if step % 100 == 0:
                    loss_value, psnr_GR2, psnr_DR2, ssim_GR2, ssim_DR2 = sess.run(
                        [vae_loss, psnr_GR, psnr_DR, ssim_GR, ssim_DR],
                        feed_dict={t_image_ph: img_an, t_org_image_ph: image_org})
                    st = time.time()
                    second_out = sess.run(decoder_DR,
                                          feed_dict={t_image_ph: img_an, t_org_image_ph: image_org})
                    run_time = time.time() - st
                    first_out = sess.run(decoder_GR,
                                         feed_dict={t_image_ph: img_an, t_org_image_ph: image_org})
                    # 模型数据
                    model_name = str(global_steps) + '.ckpt'
                    model_saver.save(sess, os.path.join(cfg_v1_1.train_summary_path, model_name))
                    ops.data_output(excel_active=excel_active, global_steps=global_steps, step=step, loss_value=loss_value, psnr_first=psnr_GR2, psnr_second=psnr_DR2,
                                    ssim_first=ssim_GR2, ssim_second=ssim_DR2, Time=run_time, epoch=cfg_v1_1.epoch, ep=ep)
                    excel.save(excel_name)
                    print('已存储第{}global_steps的实验数据到excel:{}\n'.format(global_steps, excel_name))
                    # 图像数据
                    ops.img_save_for_all(out_path=cfg_v1_1.train_out_path, global_steps=global_steps, image_size=cfg_v1_1.image_size, image_org=image_org, img_an=img_an, first_out=first_out,
                                         second_out=second_out)
                writer.add_summary(summary, global_steps)
                global_steps += 1
        writer.close()