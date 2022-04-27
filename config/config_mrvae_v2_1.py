import os
from tools import ops

########################### mrvae_v2.1 ###########################

# ================================================ 存储路径设置 =================================================
# name = 'mrvae_v1_1'
# path = 'summary/mrvae_v1_1'
# name = 'mrvae_v2_1'
name = 'mrvae_v2_1_2'
path = 'E:\WuXu\Dawn\MRVAE-Modification\summary/mrvae_v2_1'

# train_summary_path = os.path.join(path,name,'Synthetic_Lowlight_Dataset_wx/summary')
# train_out_path = os.path.join(path,name,'Synthetic_Lowlight_Dataset_wx/out')
# eval_summary_path = os.path.join(path,name, 'test_Synthetic_Lowlight_Dataset_wx/summary')
# eval_out_path = os.path.join(path,name, 'test_Synthetic_train_set4/out')
# ========================================= end =============================================

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
# 最终出版的数据
# train_summary_path = os.path.join(path,name,'train_Synthetic_train_set4/summary')
# train_out_path = os.path.join(path,name,'train_Synthetic_train_set4/out')
# eval_summary_path = os.path.join(path,name, 'test_Synthetic_train_set4/summary')
# eval_out_path = os.path.join(path,name, 'test_Synthetic_train_set4/out')

# ============================ Synthetic_Lowlight_Dataset_wx ============================
# 最终出版的数据
train_summary_path = os.path.join(path,name,'train_Synthetic_Lowlight_Dataset_wx/summary')
train_out_path = os.path.join(path,name,'train_Synthetic_Lowlight_Dataset_wx/out')
eval_summary_path = os.path.join(path,name, 'test_Synthetic_Lowlight_Dataset_wx/summary')
eval_out_path = os.path.join(path,name, 'test_Synthetic_Lowlight_Dataset_wx/out')

# ============================ gamaLinear128   QLL ============================
# train_summary_path = os.path.join(path,name,'gamaLinear128/summary')
# train_out_path = os.path.join(path,name,'gamaLinear128/out')
# eval_summary_path = os.path.join(path,name, 'test_gamaLinear128/summary')
# eval_out_path = os.path.join(path,name, 'test_gamaLinear128/out')
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
# train_dataset = 'Synthetic_train_set4' # 投稿时用的该数据集训练
train_dataset = 'Synthetic_Lowlight_Dataset_wx'
train_dataset_path = 'E:\Dataset_LL' #
eval_dataset = 'NASA'
# train_dataset_path = 'D:\WuXu\Code\Python_code\Lightlee_enhancement\data\gamatest512-train'
# train_dataset_path = 'data\Synthetic_train_set4'
# train_dataset_path = 'E:\WuXu\Dawn\MRVAE-Modification\data\LOLdataset\LOLdataset_our485'
# train_dataset_path = 'E:/766QLL/dim2rgb_imgs/gamaLinear128' # QLL
# eval_dataset_path = 'D:\WuXu\Code\Python_code\Lightlee_enhancement\data/realImages512'
# eval_dataset_path = 'D:\WuXu\Code\Python_code\Lightlee_enhancement\data\gamatest512-valid'
# eval_dataset_path = 'D:\WuXu\Code\Python_code\Lightlee_enhancement\data\LLE_testImgs'
# eval_dataset_path = 'E:\Dataset_LL\Synthetic_Lowlight_Dataset_wx'
# eval_dataset_path = 'E:\Dataset_LL\LOLdataset\eval15'
# eval_dataset_path = r'G:\WUXU\Datasets'
# eval_dataset_path = r'E:\Dataset_LL\LLTest_Set'
eval_dataset_path = r'G:\WUXU\Datasets\low_light'
ops.create_file(train_summary_path)
ops.create_file(eval_summary_path)
ops.create_file(train_out_path)
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
batch_size = 1              # 测试改成1
is_training = False
# batch_size = 4              # 测试改成1
# is_training = True
restore_model = False
ckpt = train_summary_path

# # ================================================ 超级参数设置  QLL=================================================
# latent_dim = 1024
# image_size = 512
# epoch = 6
# lr = 0.001
# lr_decay = 0.99
# lr_range = (1e-3,1e-6,0.96) #学习率退火算法（以0.96的衰减速率从1e-3到1e-6）
# lr_decay_batches = 5000     # 学习率每5000次衰减一次
# batch_size = 1             # 测试改成1
# is_training = False
# # batch_size = 4              # 测试改成1
# # is_training = True
# restore_model = False
# ckpt = train_summary_path
# # ==================================================== end =====================================================