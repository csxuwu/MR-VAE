

import os
from choice import cfg_v1_2
# from choice import cfg
from choice import cfg_v2_1
os.environ['CUDA_VISIBLE_DEVICES']='0'
if __name__ =='__main__':

    if cfg_v2_1.is_training:
        from train import trainer_for_mrvae_v2_1
        trainer_for_mrvae_v2_1.train()
    elif not cfg_v2_1.is_training:
        from eval import eval_for_mrvae_v2_1
        eval_for_mrvae_v2_1.eval()        # 真实图像
        # eval_for_mrvae_v2_1.eval2()         # 合成图像
        # eval_for_mrvae_v2_1.output_flops_and_params()
        # eval_for_mrvae_v2_1.eval2()
