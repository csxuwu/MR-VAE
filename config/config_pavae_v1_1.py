import os
from tools import ops

########################### mrvae_v2.1 ###########################

# ================================================ 存储路径设置 =================================================
# name = 'mrvae_v1_1'
# path = 'summary/mrvae_v1_1'
# name = 'mrvae_v2_1'
name = 'pavae_v1_1_1'
path = 'summary/pavae_v1_1_1'

# ========================= mrvae_v2_2_1 LOLdataset_our485 =========================
# train_summary_path = os.path.join(path,name,'train_LOLdataset_our485_2_restore/summary')
# train_out_path = os.path.join(path,name,'train_LOLdataset_our485_2_restore/out')
# eval_summary_path = os.path.join(path,name, 'test_LOLdataset_our485_2_restore/summary')
# eval_out_path = os.path.join(path,name, 'test_LOLdataset_our485_2_restore/out')
# ======================================== end ========================================

# ========================= gamatest512-train ==========================
# train_summary_path = os.path.join(path,name,'train_gamatest512-train/summary')
# train_out_path = os.path.join(path,name,'train_gamatest512-train/out')
# eval_summary_path = os.path.join(path,name, 'test_gamatest512-train/summary')
# eval_out_path = os.path.join(path,name, 'test_gamatest512-train/out')
# ======================================= end =======================================

# ============================ Synthetic_train_set4 ============================
train_summary_path = os.path.join(path,name,'train_Synthetic_train_set4/summary')
train_out_path = os.path.join(path,name,'train_Synthetic_train_set4/out')
eval_summary_path = os.path.join(path,name, 'test_Synthetic_train_set4/summary')
eval_out_path = os.path.join(path,name, 'test_Synthetic_train_set4/out')
# ========================================= end =============================================

# ============================ Synthetic_train_set3 ============================
# train_summary_path = os.path.join(path,sub_name, 'train_Synthetic_train_set3/summary')
# train_out_path = os.path.join(path,sub_name, 'train_Synthetic_train_set3/out')
# eval_summary_path = os.path.join(path, sub_name, 'test_Synthetic_train_set3/summary_test')
# eval_out_path = os.path.join(path, sub_name, 'test_Synthetic_train_set3/out_test')
# ========================================= end =============================================

# train_dataset = 'LOLdataset_our485'
# train_dataset = 'Synthetic_train_set4_continue3000_train'
# train_dataset = 'gamatest512-train'
train_dataset = 'Synthetic_train_set4'
# train_dataset = 'Synthetic_train_set3'
eval_dataset = 'LE_testImgs'
train_dataset_path = 'D:\WuXu\Code\Python_code\Lightlee_enhancement\data\gamatest512-train'
# train_dataset_path = 'data\Synthetic_train_set4'
# train_dataset_path = 'E:\WuXu\Dawn\MRVAE-Modification\data\LOLdataset\LOLdataset_our485'
# eval_dataset_path = 'D:\WuXu\Code\Python_code\Lightlee_enhancement\data/realImages512'
# eval_dataset_path = 'D:\WuXu\Code\Python_code\Lightlee_enhancement\data\gamatest512-valid'
eval_dataset_path = 'D:\WuXu\Code\Python_code\Lightlee_enhancement\data\LLE_testImgs'
ops.create_file(train_summary_path)
ops.create_file(eval_summary_path)
ops.create_file(train_out_path)
ops.create_file(eval_out_path)
# ==================================================== end =====================================================

# ================================================ 超级参数设置 =================================================
latent_dim = 1024
image_size = 512
epoch = 7
lr = 0.001
lr_decay = 0.99
lr_range = (1e-3,1e-6,0.96) #学习率退火算法（以0.96的衰减速率从1e-3到1e-6）
lr_decay_batches = 5000     # 学习率每5000次衰减一次
# batch_size = 1              # 测试改成1
# is_training = False
batch_size = 4              # 测试改成1
is_training = True
restore_model = False
ckpt = train_summary_path
# ==================================================== end =====================================================