import os
from tools import ops

########################### mrvae_v2.2 ###########################

# ================================================ 存储路径设置 =================================================
# name = 'mrvae_v2_2'
# name = 'mrvae_v2_2_3'
name = 'vae_v1_1'
path = 'summary/vae_v1'
# ========================= gamatest512-train ==========================
# train_summary_path = os.path.join(path,name,'train_gamatest512-train/summary')
# train_out_path = os.path.join(path,name,'train_gamatest512-train/out')
# eval_summary_path = os.path.join(path,name, 'test_gamatest512-train/summary2')
# eval_out_path = os.path.join(path,name, 'test_gamatest512-train/out2')
# ======================================= end =======================================

# train_summary_path = 'D:\WuXu\Code\Python_code/vae_multi09/vae_multi0903\log/vae_multi0903-07/train_epoch6_2/summary'

# ========================= LOLdataset_our485_2 =========================
# train_summary_path = os.path.join(path,'train_LOLdataset_our485_2/summary')
# train_out_path = os.path.join(path,'train_LOLdataset_our485_2/out')
# eval_summary_path = os.path.join(path, 'test_LOLdataset_our485_2/summary')
# eval_out_path = os.path.join(path, 'test_LOLdataset_our485_2/out')
# ======================================== end ========================================

# ========================= Synthetic_train_set4 =========================
train_out_path = os.path.join(path,name,'train_Synthetic_train_set4/out')
train_summary_path = os.path.join(path,name,'train_Synthetic_train_set4/summary')
eval_summary_path = os.path.join(path,name, 'test_Synthetic_train_set4/summary')
eval_out_path = os.path.join(path,name, 'test_Synthetic_train_set4/out')
# ======================================== end ========================================

# ======================================== 训练集\测试集 ========================================
train_dataset = 'Synthetic_train_set4'
# train_dataset = 'LOLdataset_our485_2'
eval_dataset = 'LLE_testImgs'
# train_dataset = 'gamatest512-train'
# train_dataset_path = 'D:\WuXu\Code\Python_code\Lightlee_enhancement\data\gamatest512-train'
# eval_dataset_path = 'D:\WuXu\Code\Python_code\Lightlee_enhancement\data/realImages512'
# eval_dataset_path = 'D:\WuXu\Code\Python_code\Lightlee_enhancement\data\gamatest512-valid'
# eval_dataset_path = 'D:\WuXu\Code\Python_code\Lightlee_enhancement\data\LLE_testImgs'
# train_dataset_path = 'E:\WuXu\Dawn\MRVAE-Modification\data\Synthetic_train_set4'
# train_dataset_path = 'E:\WuXu\Dawn\MRVAE-Modification\data\LOLdataset\LOLdataset_our485_2'
# eval_dataset_path = 'E:\WuXu\Dawn\MRVAE-Modification\data'
train_dataset_path = 'E:\WuXu\Dawn\MRVAE-Modification\data\Synthetic_train_set4'
eval_dataset_path = 'D:\WuXu\Code\Python_code\Lightlee_enhancement\data\LLE_testImgs'
# ======================================== end ========================================

ops.create_file(train_summary_path)
ops.create_file(train_out_path)
ops.create_file(eval_summary_path)
ops.create_file(eval_out_path)
# ==================================================== end =====================================================

# ================================================ 超级参数设置 =================================================
latent_dim = 1024
image_size = 512
epoch = 6
lr = 0.001
lr_decay = 0.99
lr_range = (1e-3,1e-6,0.96) #学习率退火算法（以0.96的衰减速率从1e-3到1e-6）
lr_decay_batches = 5000     # 学习率每5000次衰减一次
# batch_size = 1
# is_training = False
batch_size = 4
is_training = True
restore_model = False
ckpt = train_summary_path
# ==================================================== end =====================================================