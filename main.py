import os
from choice import cfg_v1_2
# from choice import cfg
# from choice import cfg_v2_2
from choice import cfg_v1_1
os.environ['CUDA_VISIBLE_DEVICES']='2'
if __name__ =='__main__':

    # if cfg_v1_2.is_training:
    #     from train import trainer_for_mrvae_v1_2
    #     trainer_for_mrvae_v1_2.train()
    # else:
    #     from eval import eval_for_mrvae_v1_1
    #     eval_for_mrvae_v1_1.eval_systhesis()
    if cfg_v1_1.is_training:
        from train import trainer_for_mrvae_v1_1
        trainer_for_mrvae_v1_1.train()