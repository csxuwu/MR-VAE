


import cv2
from PIL import Image


p = r'E:\WuXu\Normandy\Code\Pytorch\VAE_LL\summary\VAE_V2_1\VAE_V2_1_1\train_Systhetic_train_set4\out\0_an_0.jpg'
img1 = cv2.imread(p)
img11 = img1[:,:,(2,1,0)]
img2 = Image.open(p)

print('t')