import os
from tools import ops

########################### mrvae_v2.1 ###########################

# ================================================ 存储路径设置 =================================================
# name = 'mrvae_v1_1'
# path = 'summary/mrvae_v1_1'
# name = 'mrvae_v2_1'
name = 'mrvae_v2_1_2'
path = 'E:\WuXu\Dawn\MRVAE-Modification\summary/mrvae_v2_1' # 模型日志、数据存储路径

# ============================ Synthetic_Lowlight_Dataset_wx ============================
# 训练、测试的日志、数据存储路径
train_summary_path = os.path.join(path,name,'train_Synthetic_Lowlight_Dataset_wx/summary')
train_out_path = os.path.join(path,name,'train_Synthetic_Lowlight_Dataset_wx/out')
eval_summary_path = os.path.join(path,name, 'test_Synthetic_Lowlight_Dataset_wx/summary')
eval_out_path = os.path.join(path,name, 'test_Synthetic_Lowlight_Dataset_wx/out')

# 训练数据集
train_dataset = 'Synthetic_Lowlight_Dataset_wx'
train_dataset_path = 'E:\Dataset_LL' #

# 测试数据集
eval_dataset = 'NASA'
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
