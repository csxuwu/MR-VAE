
import cv2
import os
from tools import ops

def sum_imgs():
    path = r"E:\WuXu\Dawn\MRVAE-Modification\summary\mrvae_v2_1\mrvae_v2_1_2\test_Synthetic_train_set4\out\feature_map\encoder_conv1"

    list = os.listdir(path)
    t = []
    for i in list:
        fp = os.path.join(path,i)
        img = cv2.imread(fp)
        # img2 = img * 255
        img2 = img / 255
        t.append(img2)

    outpath = os.path.join(path,'out3.jpg')
    img2 = sum(one for one in t )
    cv2.imwrite(outpath,img2)

def multiple_img():
    path = r"E:\WuXu\Dawn\MRVAE-Modification\summary\mrvae_v2_1\mrvae_v2_1_2\test_Synthetic_train_set4\out\feature_map\fms"
    path2 = r"E:\WuXu\Dawn\MRVAE-Modification\summary\mrvae_v2_1\mrvae_v2_1_2\test_Synthetic_train_set4\out\feature_map\fms_new"
    ops.create_file(path2)

    list = os.listdir(path)
    for l in list:
        fp = os.path.join(path,l)
        img = cv2.imread(fp)
        # img2 = img * 25
        img2 = img * 15
        op = os.path.join(path2,l)
        cv2.imwrite(op,img2)

def multiple_img2():
    path = r"E:\WuXu\Dawn\MRVAE-Modification\summary\mrvae_v2_1\mrvae_v2_1_2\train_Synthetic_train_set4\out\feature_map\0"
    path2 = r"E:\WuXu\Dawn\MRVAE-Modification\summary\mrvae_v2_1\mrvae_v2_1_2\test_Synthetic_train_set4\out\feature_map\0_new"
    ops.create_file(path2)

    list = os.listdir(path)
    for l in list:
        fp = os.path.join(path,l)
        list2 = os.listdir(fp)
        for l2 in list2:
            fp2 = os.path.join(fp,l2)
            img = cv2.imread(fp2)
            img2 = img * 15
            # img2 = img * 15
            op = os.path.join(fp,l2)
            ops.create_file(op)
            cv2.imwrite(op,img2)
        print('已处理 {}'.format(l))

multiple_img2()
# sum_imgs()