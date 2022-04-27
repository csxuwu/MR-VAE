from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
from tools import ops

def plt_RGB_hist(in_path, out_path):
    for img_name in os.listdir(in_path):
        img_fullpath = os.path.join(in_path,img_name)
        img = Image.open(img_fullpath)
        r,g,b = img.split()
        ar = np.array(r).flatten()
        ag = np.array(g).flatten()
        ab = np.array(b).flatten()

        plt.hist(ar,bins=256,color='r',normed=1)
        plt.hist(ag,bins=256,color='g',normed=1)
        plt.hist(ab,bins=256,color='b',normed=1)

        hist_outpath = os.path.join(out_path,img_name)
        plt.savefig(hist_outpath)

        print('{} done.'.format(img_name))

# in_path = 'E:\WuXu\Dawn\MRVAE-Modification\data\eos_0825'
in_path = r'E:\WuXu\Dawn\MRVAE-Modification\data\hist\nor_img'
out_path = r'E:\WuXu\Dawn\MRVAE-Modification\data\hist\nor_img_hist'
ops.create_file(out_path)
plt_RGB_hist(in_path,out_path)