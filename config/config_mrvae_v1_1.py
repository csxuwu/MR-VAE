import os
from tools import ops

########################### mrvae_v1.1 ###########################

# name = 'mrvae_v1_1'
# path = 'summary/mrvae_v1_1'
name = 'mrvae_v1_1'
path = 'summary/mrvae_v1_1'

train_summary_path = os.path.join(path,'train/summary')
train_out_path = os.path.join(path,'train/out')
eval_summary_path = os.path.join(path, 'test/summary')
eval_out_path = os.path.join(path, 'test/out')

train_dataset = 'gamatest512-train'
eval_dataset = 'realImages512'
train_dataset_path = 'D:\WuXu\Code\Python_code\Lightlee_enhancement\data\gamatest512-train'
# eval_dataset_path = 'D:\WuXu\Code\Python_code\Lightlee_enhancement\data/realImages512'
# eval_dataset_path = 'D:\WuXu\Code\Python_code\Lightlee_enhancement\data\gamatest512-valid'
eval_dataset_path = 'D:\WuXu\Code\Python_code\Lightlee_enhancement\data\LLE_testImgs'
ops.create_file(train_summary_path)
ops.create_file(eval_summary_path)
ops.create_file(train_out_path)
ops.create_file(eval_out_path)

latent_dim = 1024
image_size = 512
epoch = 6
batch_size = 1
lr = 0.001
lr_decay = 0.99
is_training = True
